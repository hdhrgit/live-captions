#!/usr/bin/env python3
"""A fake ASR server for tests: the relay's HTTP contract (README, "ASR HTTP
contract") with deterministic answers and no models.

  python tests/fake_asr.py [--port 8096] [--delay 0.2] [--mode ok|error|slow|down]

Every chunk gets one segment per 3 s of audio, "chunk <n> part <k>.", timed
on the audio, so tests can check cue text and timing. --mode (or
POST /mode?set=...) switches behaviour while it runs:
  ok     answer after --delay seconds
  error  500 for every decode
  slow   answer after 30 s (longer than the relay waits)
  down   /health and decodes fail (503), as if the server were gone
"""
import argparse
import io
import json
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

STATE = {"mode": "ok", "delay": 0.2, "requests": 0, "chunks": 0}
LOCK = threading.Lock()


def segments(duration, n):
    out, t, k = [], 0.0, 1
    while t < duration - 0.5:
        end = min(duration, t + 3.0)
        out.append({"start": round(t + 0.2, 2), "end": round(end - 0.2, 2), "text": f" Chunk {n} part {k}.",
                    "no_speech_prob": 0.01, "avg_logprob": -0.2, "compression_ratio": 1.1})
        t, k = end, k + 1
    return out


class Handler(BaseHTTPRequestHandler):
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
        path = urlparse(self.path).path
        if path == "/health":
            if STATE["mode"] == "down":
                return self.reply(503, {"ok": False})
            return self.reply(200, {"ok": True, "api": 4, "mt": ["whisper", "nllb", "seamless"],
                                    "backend": "FakeASR", "model": "fake"})
        if path == "/stats":
            return self.reply(200, STATE)
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/mode":
            STATE["mode"] = q.get("set", ["ok"])[0]
            return self.reply(200, STATE)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with LOCK:
            STATE["requests"] += 1
        mode = STATE["mode"]
        if mode == "down":
            return self.reply(503, {"error": "down"})
        if mode == "error":
            return self.reply(500, {"error": "fake failure"})
        try:
            with wave.open(io.BytesIO(body)) as w:
                duration = w.getnframes() / w.getframerate()
        except (wave.Error, EOFError) as e:
            return self.reply(400, {"error": f"invalid WAV: {e}"})
        time.sleep(30 if mode == "slow" else STATE["delay"])
        with LOCK:
            STATE["chunks"] += 1
            n = STATE["chunks"]
        segs = segments(duration, n)
        lang = q.get("language", ["auto"])[0]
        common = {"speech_ratio": 0.9, "language": "en" if lang == "auto" else lang,
                  "language_probability": 0.99, "duration": duration, "processing_time": STATE["delay"]}
        if url.path == "/v1/multi":
            tracks = q.get("tracks", ["whisper"])[0].split(",")
            return self.reply(200, {"tracks": {t: segs for t in tracks}, "errors": {}, **common})
        if url.path == "/v1/translate":
            return self.reply(200, {"segments": segs, "mt": q.get("mt", ["whisper"])[0], **common})
        self.reply(404, {"error": "not found"})


def serve(port, delay=0.2, mode="ok"):
    """Start in a thread (for tests); returns the server."""
    STATE.update(delay=delay, mode=mode)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--delay", type=float, default=0.2)
    ap.add_argument("--mode", default="ok")
    a = ap.parse_args()
    STATE.update(delay=a.delay, mode=a.mode)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    srv.daemon_threads = True
    print(f"fake ASR on :{a.port} ({a.mode}, {a.delay} s)", flush=True)
    srv.serve_forever()
