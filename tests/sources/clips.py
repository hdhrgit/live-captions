"""Synthetic sources for the relay tests: a moving test pattern with a tone,
encoded in broadcast-like formats, served as live HLS or as a plain MPEG-TS
stream from a local HTTP server.

  make_clip(fmt, seconds, path)  -> path (cached)
  Source(fmt).start()            -> .hls_url / .ts_url; .stop_for(seconds) drops it
"""
import io
import os
import threading
import time
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import av
import numpy as np

CACHE = os.environ.get("TEST_CLIPS", "/tmp/captions-dvr-clips")
SEG = 2.0

# name: (video codec, width, height, frame rate, interlaced, audio codec, layout)
FORMATS = {
    "h264_25p_aac": ("libx264", 640, 360, Fraction(25), False, "aac", "stereo"),
    "h264_2997p_aac": ("libx264", 640, 360, Fraction(30000, 1001), False, "aac", "stereo"),
    "h264_50p_aac": ("libx264", 640, 360, Fraction(50), False, "aac", "stereo"),
    "h264_5994p_ac3": ("libx264", 640, 360, Fraction(60000, 1001), False, "ac3", "5.1"),
    "mpeg2_2997i_ac3": ("mpeg2video", 1920, 1080, Fraction(30000, 1001), True, "ac3", "5.1"),
    "mpeg2_25i_mp2": ("mpeg2video", 720, 576, Fraction(25), True, "mp2", "stereo"),
    "hevc_2997p_aac": ("libx265", 640, 360, Fraction(30000, 1001), False, "aac", "stereo"),
}


def make_clip(name, seconds=60):
    """An MPEG-TS clip of FORMATS[name], a keyframe every SEG seconds."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}_{seconds}s.ts")
    if os.path.exists(path):
        return path
    vcodec, w, h, rate, interlaced, acodec, layout = FORMATS[name]
    out = av.open(path + ".tmp", "w", format="mpegts")
    v = out.add_stream(vcodec, rate=rate)
    v.width, v.height, v.pix_fmt = w, h, "yuv420p"
    gop = str(round(float(rate) * SEG))
    if vcodec == "libx264":
        v.options = {"preset": "ultrafast", "g": gop, "keyint_min": gop, "sc_threshold": "0", "bf": "2"}
    elif vcodec == "libx265":
        v.options = {"preset": "ultrafast", "x265-params": f"keyint={gop}:min-keyint={gop}:scenecut=0:log-level=error"}
    else:
        v.options = {"g": gop, "b": "2000k"}
        if interlaced:
            v.codec_context.flags |= av.codec.context.Flags.interlaced_dct | av.codec.context.Flags.interlaced_me
            v.options["top"] = "1"
    a = out.add_stream(acodec, rate=48000, layout=layout)
    channels = 6 if layout == "5.1" else 2
    fsize = a.codec_context.frame_size or 1152
    samples, frames = 0, int(seconds * float(rate))
    rng = np.random.default_rng(1)
    for i in range(frames):
        img = np.zeros((h, w, 3), np.uint8)
        x = int((i * 8) % w)
        img[:, max(0, x - 20):x + 20] = (230, 230, 230)
        img[h // 3:h // 3 + 20, :, 1] = (i * 5) % 256
        vf = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
        vf.pts = i
        for p in v.encode(vf):
            out.mux(p)
        while samples < (i + 1) * 48000 / float(rate):
            # Speech-like: noise in 4 Hz "syllables", with a pause every 3 s
            # (a steady tone is dropped by the relay's gate, as bars would be).
            t = (np.arange(fsize) + samples) / 48000
            env = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) * ((t % 3.0) < 2.4)
            voice = (0.3 * env * rng.standard_normal(fsize)).astype(np.float32)
            af = av.AudioFrame.from_ndarray(np.tile(voice, (channels, 1)), format="fltp", layout=layout)
            af.sample_rate, af.pts = 48000, samples
            for p in a.encode(af):
                out.mux(p)
            samples += fsize
    for s in (v, a):
        for p in s.encode(None):
            out.mux(p)
    out.close()
    os.replace(path + ".tmp", path)
    return path


def split(path):
    """[(duration, ts bytes)]: the clip cut at keyframes into SEG-second segments."""
    src = av.open(path)
    vs, aus = src.streams.video[0], src.streams.audio[0]
    segs, buf, out, streams, start, last = [], None, None, {}, None, None

    def close():
        if out is not None:
            out.close()
            segs.append((round(last - start, 3), buf.getvalue()))
    for pkt in src.demux(vs, aus):
        if pkt.dts is None:
            continue
        t = float(pkt.pts * pkt.time_base) if pkt.pts is not None else float(pkt.dts * pkt.time_base)
        if pkt.stream is vs and pkt.is_keyframe and (start is None or t - start >= SEG - 0.01):
            if out is not None:
                last = t
                close()
            buf = io.BytesIO()
            out = av.open(buf, "w", format="mpegts")
            streams = {vs: out.add_stream_from_template(vs), aus: out.add_stream_from_template(aus)}
            start = t
        if out is None:
            continue
        pkt.stream = streams[pkt.stream]
        out.mux(pkt)
        last = max(last or t, t)
    close()
    return segs


class Source:
    """A live source over HTTP: /live.m3u8 (a sliding window over the clip's
    segments, published in real time) and /live.ts (the clip at real-time pace)."""

    def __init__(self, name, seconds=60):
        self.path = make_clip(name, seconds)
        self.segs = split(self.path)
        self.t0 = None
        self.down_until = 0.0

    def start(self, backlog=3):
        src = self
        # Start a few segments in, so the playlist has a backlog like a real channel.
        self.t0 = time.monotonic() - backlog * SEG

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if time.monotonic() < src.down_until:
                    self.send_error(503)
                    return
                if self.path.startswith("/live.m3u8"):
                    return self.reply("application/vnd.apple.mpegurl", src.playlist().encode())
                if self.path.startswith("/seg"):
                    n = int(self.path[4:].split(".")[0])
                    return self.reply("video/mp2t", src.segs[n % len(src.segs)][1])
                if self.path.startswith("/live.ts"):
                    return self.stream()
                self.send_error(404)

            def reply(self, ctype, body):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "video/mp2t")
                self.end_headers()
                data = open(src.path, "rb").read()
                rate = len(data) / sum(d for d, _ in src.segs)   # bytes per second
                t0, pos = time.monotonic(), 0
                try:
                    while pos < len(data):
                        n = int((time.monotonic() - t0 + 2) * rate) - pos   # 2 s ahead
                        if n > 0:
                            chunk = data[pos:pos + n - n % 188]
                            self.wfile.write(chunk)
                            pos += len(chunk)
                        time.sleep(0.1)
                except OSError:
                    pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{self.srv.server_port}"
        self.hls_url, self.ts_url = base + "/live.m3u8", base + "/live.ts"
        return self

    def playlist(self):
        """The segments published by now: a window of the last 6."""
        n = min(len(self.segs), int((time.monotonic() - self.t0) / SEG))
        first = max(0, n - 6)
        lines = ["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{int(SEG) + 1}",
                 f"#EXT-X-MEDIA-SEQUENCE:{first}"]
        for i in range(first, n):
            lines += [f"#EXTINF:{self.segs[i][0]:.3f},", f"seg{i}.ts"]
        return "\n".join(lines) + "\n"

    def stop_for(self, seconds):
        self.down_until = time.monotonic() + seconds

    def close(self):
        self.srv.shutdown()
