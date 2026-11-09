"""Reference ASR server implementing the captions-dvr HTTP contract.

  POST /v1/translate?language=<code|auto>&task=<translate|transcribe>&mt=<whisper|nllb|seamless>
                                   body: WAV, 16 kHz mono s16le
  -> {"segments": [{"start", "end", "text", "no_speech_prob",
                    "avg_logprob", "compression_ratio"}], "speech_ratio",
      "language", "language_probability", "mt", "duration", "processing_time"}
  GET /health -> {"ok": true, "api": 3, "mt": [...], "backend": ..., "model": ...}

ASR_BACKEND=mlx uses mlx-whisper on Apple Silicon (see mac/);
ASR_BACKEND=cuda uses faster-whisper on an NVIDIA GPU (untested, see
docs/NVIDIA.md).
"""
import io
import json
import logging
import multiprocessing as mp
import os
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
    while True:
        audio, lang, task, mt = conn.recv()
        try:
            if task == "multi":
                conn.send(("ok", m.multi(audio, lang, mt)))
            else:
                conn.send(("ok", m.transcribe(audio, lang, task, mt)))
        except Exception as e:  # noqa: BLE001 - reported to the client
            conn.send(("err", f"{type(e).__name__}: {e}"))


class Worker:
    def __init__(self, kind="main"):
        self.kind = kind
        self.gate = threading.Lock()   # one decode at a time per worker
        self.ctx = mp.get_context("spawn")
        self.spawn()

    def spawn(self):
        self.conn, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_worker_main, args=(child, self.kind), daemon=True)
        self.proc.start()
        self.ready = False
        logging.info("worker %d starting", self.proc.pid)

    def wait_ready(self, timeout):
        if not self.ready and self.conn.poll(timeout):
            self.ready = self.conn.recv() == "ready"
            logging.info("worker %d ready", self.proc.pid)
        return self.ready

    def transcribe(self, audio, lang, task, mt, deadline):
        self.conn.send((audio, lang, task, mt))
        if self.conn.poll(deadline):
            status, r = self.conn.recv()
            if status == "ok":
                return r
            raise RuntimeError(r)
        logging.warning("decode exceeded %.0fs; killing worker %d", deadline, self.proc.pid)
        self.proc.kill()
        self.proc.join()
        self.spawn()
        raise TimeoutError(f"decode exceeded {deadline:.0f}s")


workers = {}   # "main" and, if installed, "seamless"


def read_wav(handler):
    data = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
    with wave.open(io.BytesIO(data)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (asr.SR, 1, 2):
            raise ValueError("need 16 kHz mono s16le WAV")
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, "<i2").astype(np.float32) / 32768


def run_on(worker, audio, lang, task, mt, deadline, t_in):
    """Run one job on a worker under its gate; returns (result, error)."""
    if not worker.gate.acquire(timeout=max(0.0, MAX_WAIT - (time.monotonic() - t_in))):
        return None, "busy"
    try:
        if not worker.wait_ready(max(0.0, MAX_WAIT - (time.monotonic() - t_in))):
            return None, "model loading"
        return worker.transcribe(audio, lang, task, mt, deadline), None
    except (TimeoutError, RuntimeError) as e:
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
        if urlparse(self.path).path == "/health":
            nllb_ok = os.path.exists(os.path.join(
                os.environ.get("NLLB_MODEL", "/models/nllb-200-distilled-1.3B-ct2"), "model.bin"))
            seamless_ok = os.path.exists(os.path.join(asr.seamless_path(), "config.json"))
            return self.reply(200, {"ok": True, "api": 4,
                                    "mt": ["whisper"] + (["nllb"] if nllb_ok else []) + (["seamless"] if seamless_ok else []),
                                    "backend": type(model).__name__,
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
        t_in = time.monotonic()
        worker = workers["seamless" if mt == "seamless" and "seamless" in workers else "main"]
        gate = worker.gate
        if not gate.acquire(timeout=MAX_WAIT):
            logging.warning("busy: rejected %.1fs chunk after %.0fs wait", len(audio) / asr.SR, MAX_WAIT)
            return self.reply(503, {"error": "busy"})
        try:
            if not worker.wait_ready(max(0.0, MAX_WAIT - (time.monotonic() - t_in))):
                return self.reply(503, {"error": "model loading"})
            t0 = time.monotonic()
            deadline = min(MAX_DECODE, 3 * len(audio) / asr.SR + 5)
            try:
                r = worker.transcribe(audio, lang, task, mt, deadline)
            except TimeoutError as e:
                return self.reply(503, {"error": str(e)})
            except RuntimeError as e:
                return self.reply(500, {"error": str(e)})
            el = time.monotonic() - t0
        finally:
            gate.release()
        logging.info("%.1fs audio in %.2fs, speech %s, %d segments", len(audio) / asr.SR, el,
                     r["speech_ratio"], len(r["segments"]))
        self.reply(200, {**r, "duration": len(audio) / asr.SR, "processing_time": round(el, 3)})


if __name__ == "__main__":
    workers["main"] = Worker()
    if os.path.exists(os.path.join(asr.seamless_path(), "config.json")):
        workers["seamless"] = Worker("seamless")
    srv = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8096"))), Handler)
    srv.daemon_threads = True
    srv.serve_forever()
