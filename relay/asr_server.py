"""Reference ASR server implementing the captions-dvr HTTP contract.

  POST /v1/translate?language=<code|auto>&task=<translate|transcribe>&mt=<whisper|nllb|seamless>
                                   body: WAV, 16 kHz mono s16le
  -> {"segments": [{"start", "end", "text", "no_speech_prob",
                    "avg_logprob", "compression_ratio"}], "speech_ratio",
      "language", "language_probability", "mt", "duration", "processing_time"}
  GET /health -> {"ok": true, "api": 3, "mt": [...], "backend": ..., "model": ...}

ASR_BACKEND=mlx uses mlx-whisper on Apple Silicon (see mac/);
ASR_BACKEND=cuda uses faster-whisper on an NVIDIA GPU (tested on an RTX 3060, see
docs/NVIDIA.md).
"""
import collections
import io
import json
import logging
import multiprocessing as mp
import os
import subprocess
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import asr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
model = asr.from_env()  # for /health only; decoding happens in the worker
# One decode at a time per worker. A request that cannot start within
# MAX_WAIT seconds gets 503 at once: its client (timeout ~10 s) would have
# given up anyway, and decoding it would only delay the requests behind it.
MAX_WAIT = float(os.environ.get("MAX_WAIT", "5"))
# Hard limit per decode; a worker that overruns is killed and respawned.
MAX_DECODE = float(os.environ.get("MAX_DECODE", "30"))
# Decodes a worker runs side by side on its one loaded model (threads in the
# worker process). 1 for the Mac; an NVIDIA GPU has room for several.
PARALLEL = max(1, int(os.environ.get("ASR_PARALLEL", "1")))
# Seamless requests at once: they are decoded in batches (asr.Batcher), so
# more of them cost little extra GPU time.
SEAMLESS_PARALLEL = max(1, int(os.environ.get("SEAMLESS_PARALLEL", str(PARALLEL))))


def _worker_main(conn, kind):
    """Decode loop in a child process, so a runaway decode can be killed.

    kind "main" serves Whisper (+ NLLB); kind "seamless" only Seamless, so
    a slow Seamless decode can never take the main worker down with it."""
    m = asr.from_env()
    if kind == "seamless":
        asr.seamless().translate(np.zeros(asr.SR, np.float32))  # load before taking requests
    else:
        m.transcribe(np.zeros(asr.SR, np.float32), "en", "transcribe")  # load + warm up
        if asr.nllb():
            asr.nllb().translate(["안녕하세요"], "ko")
    conn.send("ready")
    send_lock = threading.Lock()

    def run(rid, audio, lang, task, mt):
        try:
            r = ("ok", m.multi(audio, lang, mt) if task == "multi" else m.transcribe(audio, lang, task, mt))
        except Exception as e:  # noqa: BLE001 - reported to the client
            r = ("err", f"{type(e).__name__}: {e}")
        with send_lock:
            conn.send((rid,) + r + (dict(asr.BATCHES),))

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(SEAMLESS_PARALLEL if kind == "seamless" else PARALLEL)
    while True:
        rid, audio, lang, task, mt = conn.recv()
        pool.submit(run, rid, audio, lang, task, mt)


class Worker:
    def __init__(self, kind="main"):
        self.kind = kind
        self.limit = SEAMLESS_PARALLEL if kind == "seamless" else PARALLEL
        self.gate = threading.BoundedSemaphore(self.limit)
        self.ctx = mp.get_context("spawn")
        self.lock = threading.Lock()      # sending, respawning, the pending table
        self.pending = {}                 # request id -> [Event, result]
        self.batches = {}                 # the worker's asr.BATCHES, as of its last reply
        self.rid = 0
        self.spawn()

    def spawn(self):
        self.conn, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_worker_main, args=(child, self.kind), daemon=True)
        self.proc.start()
        self.ready = threading.Event()
        threading.Thread(target=self.receive, args=(self.conn, self.ready), daemon=True).start()
        logging.info("worker %d starting", self.proc.pid)

    def receive(self, conn, ready):
        """Results from the worker process, handed to whoever waits for them."""
        try:
            while True:
                msg = conn.recv()
                if msg == "ready":
                    ready.set()
                    logging.info("worker %d ready", self.proc.pid)
                    continue
                rid, status, r, batches = msg
                self.batches = batches
                with self.lock:
                    slot = self.pending.pop(rid, None)
                if slot:
                    slot[1] = (status, r)
                    slot[0].set()
        except (EOFError, OSError):
            return   # the worker was killed (respawned)

    def wait_ready(self, timeout):
        return self.ready.wait(timeout)

    def transcribe(self, audio, lang, task, mt, deadline):
        slot = [threading.Event(), None]
        with self.lock:
            self.rid += 1
            rid = self.rid
            self.pending[rid] = slot
            self.conn.send((rid, audio, lang, task, mt))
        if slot[0].wait(deadline):
            status, r = slot[1]
            if status == "ok":
                return r
            raise RuntimeError(r)
        with self.lock:
            if self.pending.pop(rid, None) is None and slot[1]:   # finished just now
                status, r = slot[1]
                if status == "ok":
                    return r
            logging.warning("decode exceeded %.0fs; killing worker %d", deadline, self.proc.pid)
            self.proc.kill()
            self.proc.join()
            failed, self.pending = self.pending, {}
            self.spawn()
        for s in failed.values():   # the other decodes in that process are gone too
            s[1] = ("err", "worker restarted")
            s[0].set()
        raise TimeoutError(f"decode exceeded {deadline:.0f}s")


workers = {}   # "main" and, if installed, "seamless"


class Stats:
    """Counters for GET /metrics (Prometheus text format)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.c = collections.Counter()   # (metric, labels tuple) -> value

    def add(self, name, value=1, **labels):
        with self.lock:
            self.c[(name, tuple(sorted(labels.items())))] += value

    def text(self):
        lines, typed = [], set()
        with self.lock:
            items = sorted(self.c.items())
        for (name, labels), v in items:
            if name not in typed:
                typed.add(name)
                lines.append(f"# TYPE {name} counter")
            lab = ",".join(f'{k}="{x}"' for k, x in labels)
            lines.append(f"{name}{{{lab}}} {round(v, 3)}" if lab else f"{name} {round(v, 3)}")
        lines.append("# TYPE asr_batches_total counter")
        for w in workers.values():
            for (model, size), n in sorted(w.batches.items()):
                lines.append(f'asr_batches_total{{model="{model}",size="{size}"}} {n}')
        lines.append("# TYPE asr_in_flight gauge")
        for kind, w in workers.items():
            lines.append(f'asr_in_flight{{worker="{kind}"}} {w.limit - w.gate._value}')
        for name, value in gpu().items():
            lines += [f"# TYPE asr_gpu_{name} gauge", f"asr_gpu_{name} {value}"]
        return "\n".join(lines) + "\n"


stats = Stats()
_gpu = {"at": 0.0, "value": {}}


def gpu():
    """Utilization and memory from nvidia-smi (cached 5 s); {} without one."""
    if time.monotonic() - _gpu["at"] > 5:
        _gpu["at"], v = time.monotonic(), {}
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=3).stdout.split(",")
            v = {"utilization_percent": float(out[0]), "memory_used_bytes": float(out[1]) * 2 ** 20,
                 "memory_total_bytes": float(out[2]) * 2 ** 20}
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
        _gpu["value"] = v
    return _gpu["value"]


def read_wav(handler):
    data = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
    with wave.open(io.BytesIO(data)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (asr.SR, 1, 2):
            raise ValueError("need 16 kHz mono s16le WAV")
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, "<i2").astype(np.float32) / 32768


def run_on(worker, audio, lang, task, mt, deadline, t_in):
    """Run one job on a worker under its gate; returns (result, error)."""
    model = mt if isinstance(mt, str) else "multi"
    if not worker.gate.acquire(timeout=max(0.0, MAX_WAIT - (time.monotonic() - t_in))):
        stats.add("asr_requests_total", model=model, status="busy")
        return None, "busy"
    try:
        if not worker.wait_ready(max(0.0, MAX_WAIT - (time.monotonic() - t_in))):
            stats.add("asr_requests_total", model=model, status="loading")
            return None, "model loading"
        t0 = time.monotonic()
        stats.add("asr_queue_wait_seconds_sum", t0 - t_in, model=model)
        r = worker.transcribe(audio, lang, task, mt, deadline)
        stats.add("asr_decode_seconds_sum", time.monotonic() - t0, model=model)
        stats.add("asr_decode_seconds_count", model=model)
        stats.add("asr_audio_seconds_sum", len(audio) / asr.SR, model=model)
        stats.add("asr_requests_total", model=model, status="ok")
        return r, None
    except (TimeoutError, RuntimeError) as e:
        stats.add("asr_requests_total", model=model, status="timeout" if isinstance(e, TimeoutError) else "error")
        return None, str(e)
    finally:
        worker.gate.release()


class Handler(BaseHTTPRequestHandler):
    def multi(self, url):
        """POST /v1/multi?language=<code|auto>&tracks=nllb,whisper,original,seamless
        -> {"tracks": {name: segments}, "speech_ratio", "language", "errors", "processing_time"}.
        The main lane (Whisper transcribe -> NLLB on CPU || Whisper translate on
        GPU) decides the response time; Seamless runs in its own worker at the
        same time and is included if it has finished by then."""
        q = parse_qs(url.query)
        lang = q.get("language", ["auto"])[0]
        lang = None if lang == "auto" else lang
        tracks = [t for t in q.get("tracks", ["nllb,whisper,original"])[0].split(",") if t]
        try:
            audio = read_wav(self)
        except (ValueError, wave.Error, EOFError) as e:
            return self.reply(400, {"error": f"invalid WAV: {e!r}"})
        t_in = t0 = time.monotonic()
        deadline = min(MAX_DECODE, 3 * len(audio) / asr.SR + 5)
        side = {}
        thread = None
        if "seamless" in tracks and "seamless" in workers:
            def run_seamless():
                side["r"], side["err"] = run_on(workers["seamless"], audio, lang, "translate",
                                                "seamless", deadline, t_in)
            thread = threading.Thread(target=run_seamless, daemon=True)
            thread.start()
        main_tracks = [t for t in tracks if t != "seamless"]
        r, err = run_on(workers["main"], audio, lang, "multi", main_tracks, deadline, t_in)
        if r is None:
            return self.reply(503, {"error": err})
        out = {"tracks": r["tracks"], "speech_ratio": r["speech_ratio"], "language": r["language"],
               "errors": {}}
        if thread is not None:
            thread.join(timeout=0.5)       # never hold the main result for Seamless
            if side.get("r"):
                out["tracks"]["seamless"] = side["r"]["segments"]
            else:
                out["errors"]["seamless"] = side.get("err") or "not ready"
        out["processing_time"] = round(time.monotonic() - t0, 3)
        logging.info("multi %.1fs audio in %.2fs: %s", len(audio) / asr.SR, out["processing_time"],
                     {k: len(v) for k, v in out["tracks"].items()})
        self.reply(200, out)


    def log_message(self, fmt, *args):
        pass

    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/metrics":
            body = stats.text().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        if urlparse(self.path).path == "/health":
            nllb_ok = os.path.exists(os.path.join(
                os.environ.get("NLLB_MODEL", "/models/nllb-200-distilled-1.3B-ct2"), "model.bin"))
            seamless_ok = "seamless" in workers
            return self.reply(200, {"ok": True, "api": 4,
                                    "mt": ["whisper"] + (["nllb"] if nllb_ok else []) + (["seamless"] if seamless_ok else []),
                                    "backend": type(model).__name__,
                                    "parallel": {"main": PARALLEL, "seamless": SEAMLESS_PARALLEL},
                                    "batch": {"whisper": asr.batch_size("whisper"), "seamless": asr.batch_size("seamless")},
                                    "model": getattr(model, "name", None) or getattr(model, "model", None)})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/v1/multi":
            return self.multi(url)
        if url.path != "/v1/translate":
            return self.reply(404, {"error": "not found"})
        q = parse_qs(url.query)
        lang = q.get("language", ["ko"])[0]
        lang = None if lang == "auto" else lang
        task = q.get("task", ["translate"])[0]
        mt = q.get("mt", ["whisper"])[0]
        if task not in ("translate", "transcribe"):
            return self.reply(400, {"error": "task must be translate or transcribe"})
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            with wave.open(io.BytesIO(data)) as w:
                if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (asr.SR, 1, 2):
                    return self.reply(400, {"error": "need 16 kHz mono s16le WAV"})
                pcm = w.readframes(w.getnframes())
        except (wave.Error, EOFError) as e:
            return self.reply(400, {"error": f"invalid WAV: {e!r}"})
        audio = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768
        t_in = t0 = time.monotonic()
        worker = workers["seamless" if mt == "seamless" and "seamless" in workers else "main"]
        r, err = run_on(worker, audio, lang, task, mt, min(MAX_DECODE, 3 * len(audio) / asr.SR + 5), t_in)
        el = time.monotonic() - t0
        if r is None:
            if err == "busy":
                logging.warning("busy: rejected %.1fs chunk after %.0fs wait", len(audio) / asr.SR, MAX_WAIT)
            return self.reply(500 if err not in ("busy", "model loading") and not err.startswith("decode exceeded")
                              else 503, {"error": err})
        logging.info("%.1fs audio in %.2fs, speech %s, %d segments", len(audio) / asr.SR, el,
                     r["speech_ratio"], len(r["segments"]))
        self.reply(200, {**r, "duration": len(audio) / asr.SR, "processing_time": round(el, 3)})


def check_cuda_libs():
    """With ASR_BACKEND=cuda: exit with instructions if CTranslate2 (Whisper)
    won't find the CUDA libraries. They come with torch's pip wheels but are
    not on the loader path by default, and the first decode would fail."""
    import ctypes
    missing = []
    for lib in ("libcublas.so.12", "libcudnn.so.9"):
        try:
            ctypes.CDLL(lib)
        except OSError:
            missing.append(lib)
    if missing:
        raise SystemExit(
            f"CUDA libraries not found: {', '.join(missing)}. They come with torch's pip packages; put them on\n"
            "the loader path before starting (docs/NVIDIA.md):\n"
            '  SP=$(python -c "import site; print(site.getsitepackages()[0])")\n'
            '  export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr "\\n" ":")$LD_LIBRARY_PATH')


if __name__ == "__main__":
    if os.environ.get("ASR_BACKEND") == "cuda":
        check_cuda_libs()
    workers["main"] = Worker()
    # Seamless is far too slow on a CPU for live captions: Whisper and NLLB only.
    if os.environ.get("ASR_BACKEND") != "cpu" and os.path.exists(os.path.join(asr.seamless_path(), "config.json")):
        workers["seamless"] = Worker("seamless")
    srv = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8096"))), Handler)
    srv.daemon_threads = True
    srv.serve_forever()
