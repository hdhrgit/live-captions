"""Burned-in subtitles: the "compare" mode of a channel, showing the chosen
translation backends' subtitles at once, for comparing the models.

A Burner is a video source for hls.Segmenter, like the Channel itself (the
segmenter switches between them when the channel's Compare option changes).
While it has clients it
reads the channel's own TS output (already delayed, so the cues exist
before their frames), decodes the video, draws each track's current cue
in its own band (own colour and label, stacked at the bottom), encodes
H.264 (ENCODER) and copies the audio. It reuses the channel's
captioning session, so it costs no extra ASR.
"""
import collections
import io
import logging
import os
import queue
import threading
import time

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

import hls

log = logging.getLogger("burn")

# Encoder for compare mode. Only libx264 (software) so far; a
# hardware one (h264_vaapi / h264_qsv, e.g. on the J4125 at .195) goes in
# ENCODERS with its options, and BURN_ENCODER selects it.
ENCODER = os.environ.get("BURN_ENCODER", "libx264")
ENCODERS = {
    "libx264": lambda rate: {"preset": os.environ.get("BURN_PRESET", "veryfast"), "crf": "23",
                             "maxrate": "4000k", "bufsize": "8000k", "g": str(int(float(rate) * 2)),
                             "profile": "high", "level": "4.0",
                             "threads": os.environ.get("BURN_THREADS", "4")},
}
FONT_FILES = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf")


class Restart(Exception):
    pass


class QueueReader(io.RawIOBase):
    """The channel's TS chunks as a file for av.open."""

    def __init__(self, cq, stop):
        self.cq, self.stop, self.buf = cq, stop, b""

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            if self.stop.is_set():
                return 0
            try:
                chunk = self.cq.get(timeout=5)
            except queue.Empty:
                continue
            if chunk is None:
                return 0
            self.buf = chunk
        n = min(len(b), len(self.buf))
        b[:n], self.buf = self.buf[:n], self.buf[n:]
        return n


def font(size):
    for f in FONT_FILES:
        if os.path.exists(f):
            # Raqm (with FriBiDi) orders right-to-left text; without it
            # Pillow draws Hebrew/Arabic backwards.
            layout = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
            return ImageFont.truetype(f, size, layout_engine=layout)
    return ImageFont.load_default(size)   # Pillow's bundled font (no Hebrew)


class Overlay:
    """A cue rendered once, as YUV planes plus alpha, for blending into frames."""

    def __init__(self, text, width, height, color=(255, 255, 255), slot=0):
        """slot: band number from the bottom; each band has room for 2 lines."""
        size = max(16, height * 38 // 1080)
        lines = [line for line in text.split("\n") if line.strip()][:2] or [text]
        f = font(size)
        while size > 16:
            if max(f.getlength(line) for line in lines) <= width * 0.9:
                break
            size -= 2
            f = font(size)
        pad, gap = size // 3, size // 5
        line_h = size + gap
        box_w = int(max(f.getlength(line) for line in lines)) + 2 * pad
        box_h = line_h * len(lines) + 2 * pad - gap
        box_w += box_w % 2
        box_h += box_h % 2
        img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 170))
        d = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            x = (box_w - f.getlength(line)) / 2
            d.text((x, pad + i * line_h), line, font=f, fill=color + (255,),
                   stroke_width=max(1, size // 16), stroke_fill=(0, 0, 0, 255))
        rgba = np.asarray(img).astype(np.float32)
        r, g, b, a = (rgba[..., i] for i in range(4))
        self.y = 16 + 0.257 * r + 0.504 * g + 0.098 * b      # BT.601 limited range
        u = 128 - 0.148 * r - 0.291 * g + 0.439 * b
        v = 128 + 0.439 * r - 0.368 * g - 0.071 * b
        self.a = a / 255
        half = lambda p: p.reshape(box_h // 2, 2, box_w // 2, 2).mean(axis=(1, 3))  # noqa: E731
        self.u, self.v, self.a2 = half(u * self.a), half(v * self.a), half(self.a)
        self.w, self.h = box_w, box_h
        # Centred; each band's bottom edge at a fixed height, so bands don't
        # move when another one changes. Band 0 sits above the lower edge.
        band = 2 * line_h + 2 * pad
        bottom = height - height * 5 // 100 - slot * band
        self.x = (width - box_w) // 2 // 2 * 2
        self.top = (bottom - box_h) // 2 * 2

    def blend(self, arr, width, height):
        """arr: yuv420p frame as returned by VideoFrame.to_ndarray()."""
        x, t, w, h = self.x, self.top, self.w, self.h
        y = arr[t:t + h, x:x + w].astype(np.float32)
        arr[t:t + h, x:x + w] = (y * (1 - self.a) + self.y * self.a).astype(np.uint8)
        # U and V planes follow Y, each (height/2 x width/2), stored row-packed.
        uv = arr[height:].reshape(2, height // 2, width // 2)
        for plane, src in ((0, self.u), (1, self.v)):
            p = uv[plane, t // 2:t // 2 + h // 2, x // 2:x // 2 + w // 2].astype(np.float32)
            uv[plane, t // 2:t // 2 + h // 2, x // 2:x // 2 + w // 2] = \
                (p * (1 - self.a2) + src).astype(np.uint8)


class Burner:
    """A channel's compare-mode video; subscribe/unsubscribe like a Channel."""

    def __init__(self, ch, tracks, broadcaster):
        """tracks: callable -> [(track, label, (r, g, b))], bottom band first
        (read per frame, so changes in the web page apply at once)."""
        self.ch = ch
        self.tracks = tracks
        self.id = f"{ch.id}-compare"
        self.broadcaster = broadcaster
        self.lock = threading.Lock()
        self.thread = None
        self.stop = threading.Event()
        self.overlays = collections.OrderedDict()
        self.stats = {"state": "idle"}

    cfg = property(lambda self: self.ch.cfg)
    height = property(lambda self: int(self.ch.cfg.get("burn_height") or hls.SETTINGS.get("burn_height") or 1080))

    def subscribe(self):
        cq = queue.Queue(maxsize=4000)
        cq.waiting = True
        with self.lock:
            with self.broadcaster.lock:
                self.broadcaster.clients.add(cq)
            if self.thread is None or not self.thread.is_alive():
                self.stop = threading.Event()
                self.thread = threading.Thread(target=self.run, args=(self.stop,),
                                               name=f"{self.id}", daemon=True)
                self.thread.start()
        return cq

    def unsubscribe(self, cq):
        with self.lock:
            with self.broadcaster.lock:
                self.broadcaster.clients.discard(cq)
                if not self.broadcaster.clients:
                    self.stop.set()

    def run(self, stop):
        log.info("%s: burning in %s at %dp with %s", self.id,
                 ", ".join(t for t, _, _ in self.tracks()), self.height, ENCODER)
        while not stop.is_set():
            cq = self.ch.subscribe()
            try:
                self.pipeline(cq, stop)
            except Restart as e:
                log.info("%s: restarting encoder: %s", self.id, e)
            except Exception as e:  # noqa: BLE001 - keep the variant alive
                log.warning("%s: %s: %s", self.id, type(e).__name__, e)
                time.sleep(1)
            finally:
                self.ch.unsubscribe(cq)
            self.broadcaster.reset()
        self.stats = {"state": "idle"}
        log.info("%s: stopped", self.id)

    def cue(self, track, pts):
        """The cue (text) of a track showing at this video PTS, or None."""
        for _ in range(3):
            try:
                cues = list(self.ch.vtt_tracks.get(track, ()))
                break
            except RuntimeError:   # deque appended to while copying
                continue
        else:
            return None
        for cs, ce, text in reversed(cues):
            if hls.wrap_diff(pts, cs + hls.MUX_DELAY) >= 0:
                return text if hls.wrap_diff(pts, ce + hls.MUX_DELAY) < 0 else None
        return None

    def overlay(self, text, w, h, color, slot):
        key = (text, w, h, slot)
        o = self.overlays.get(key)
        if o is None:
            o = self.overlays[key] = Overlay(text, w, h, color, slot)
            while len(self.overlays) > 30:
                self.overlays.popitem(last=False)
        return o

    def pipeline(self, cq, stop):
        src = av.open(QueueReader(cq, stop), format="mpegts", mode="r")
        try:
            vin = src.streams.video[0]
            ain = src.streams.audio[0] if src.streams.audio else None
            vin.thread_type = "AUTO"
            out = av.open(self.broadcaster, mode="w", format="mpegts")
            rate = vin.average_rate or 25
            h = self.height
            w = int(round(vin.codec_context.width * h / vin.codec_context.height / 2)) * 2 \
                if vin.codec_context.height else h * 16 // 9
            if vin.codec_context.height and vin.codec_context.height <= h:
                w, h = vin.codec_context.width, vin.codec_context.height
            # Video first: it gets PID 0x100, which hls.Segmenter cuts on.
            vout = out.add_stream(ENCODER, rate=rate)
            vout.width, vout.height, vout.pix_fmt = w, h, "yuv420p"
            vout.time_base = vin.time_base
            vout.codec_context.time_base = vin.time_base
            vout.options = ENCODERS[ENCODER](rate)
            aout = out.add_stream_from_template(ain) if ain else None
            last, frames, t0, cpu0 = None, 0, time.monotonic(), time.process_time()
            self.stats = {"state": "starting", "size": f"{w}x{h}"}
            streams = [vin] + ([ain] if ain else [])
            for pkt in src.demux(*streams):
                if stop.is_set():
                    break
                if pkt.stream is ain:
                    if pkt.dts is None:
                        continue
                    pkt.stream = aout
                    pkt.pts -= hls.MUX_DELAY
                    pkt.dts -= hls.MUX_DELAY
                    out.mux(pkt)
                    continue
                for frame in pkt.decode():
                    if frame.pts is None:
                        continue
                    if last is not None and not 0 < hls.wrap_diff(frame.pts, last) < 10 * 90000:
                        raise Restart("timestamps jumped")
                    last = frame.pts
                    arr = frame.reformat(width=w, height=h, format="yuv420p").to_ndarray()
                    tracks = self.tracks()
                    for slot, (track, label, color) in enumerate(tracks):
                        text = self.cue(track, frame.pts)
                        if text:
                            self.overlay(f"{label}: {text}", w, h, color, slot).blend(arr, w, h)
                    of = av.VideoFrame.from_ndarray(arr, format="yuv420p")
                    # The muxer adds its delay again; take it off so the
                    # stream keeps the passthrough's timing (and WebVTT cues fit).
                    of.pts, of.time_base = frame.pts - hls.MUX_DELAY, vin.time_base
                    for p in vout.encode(of):
                        out.mux(p)
                    frames += 1
                    if time.monotonic() - t0 >= 10:
                        el = time.monotonic() - t0
                        self.stats = {"state": "encoding", "size": f"{w}x{h}",
                                      "fps": round(frames / el, 1),
                                      # whole process (all channels), for comparison only
                                      "process_cpu_pct": round(100 * (time.process_time() - cpu0) / el)}
                        frames, t0, cpu0 = 0, time.monotonic(), time.process_time()
            out.close()
        finally:
            src.close()
