"""Slate card shown while a channel starts.

A cold channel needs 6-15 s before its first real HLS segment exists
(opening the source, then a whole keyframe interval). Meanwhile the
segmenter serves 2 s segments of a generic dark card with silent audio, so
the DVR's first playlist request already has segments; the channel's name
goes in a subtitle cue over it.

The card is encoded once per source format (any channel, managed or
stateless, with that format shares it) and cached on disk:
SLATE_SECONDS of 2 s segments, timestamps from 0, which is below where the
relay's timeline starts (relay.Timeline), so the real stream continues
forward after it.
"""
import io
import json
import logging
import os
import re
import threading

import av
import numpy as np
from PIL import Image, ImageDraw

import burn
import hls

log = logging.getLogger("slate")

SEG = 2.0                 # seconds per slate segment
SLATE_SECONDS = 50        # cached length; enough for a slow source to start (below the 60 s timeline start)
CACHE_DIR = os.environ.get("SLATE_DIR", "")
DEFAULT = {"width": 1920, "height": 1080, "fps": 25, "sample_rate": 48000, "channels": 2, "profile": "High",
           "vcodec": "h264", "acodec": "aac"}
_lock = threading.Lock()
_mem = {}
_rendering = set()


def _key(fmt):
    return "slate-{width}x{height}-{fps}-{sample_rate}-{channels}-{p}{c}".format(
        p=re.sub(r"\W", "", fmt.get("profile") or "High").lower(),
        c="" if (fmt.get("vcodec"), fmt.get("acodec")) == ("h264", "aac") else f"-{fmt.get('vcodec')}-{fmt.get('acodec')}",
        **fmt)


def _formats_path():
    return os.path.join(CACHE_DIR, "formats.json") if CACHE_DIR else None


def _formats():
    try:
        with open(_formats_path()) as f:
            return json.load(f)
    except (OSError, TypeError, ValueError):
        return {}


def remember(ch_id, fmt):
    """Note a channel's source format (for its next cold start), and render its card."""
    known = _formats()
    if known.get(ch_id) != fmt and _formats_path():
        known[ch_id] = fmt
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(_formats_path(), "w") as f:
                json.dump(known, f)
        except OSError as e:
            log.warning("can't save slate formats: %s", e)
    if _cached(_key(fmt)) is None:
        threading.Thread(target=segments, args=(fmt,), name="slate", daemon=True).start()


def fmt_of(ch):
    """The video/audio format seen on this channel's source (now or on an
    earlier run), or None if it was never opened."""
    fmt = getattr(ch, "source_format", None) or _formats().get(ch.id)
    return dict(DEFAULT, **fmt) if fmt else None


def get(ch):
    """The slate in the channel's source format if it's ready; else the
    closest cached one (same resolution, else 1080p). Only an exact match is
    sure to play cleanly: a card in another resolution or encoding before the
    real picture can make Apple TV (through the DVR) show green, but a green
    frame beats a "no segments" error."""
    fmt = fmt_of(ch)
    if fmt:
        segs = _cached(_key(fmt))
        if segs:
            return segs
    if fmt is None and ".m3u" not in (ch.cfg.get("url") or "").split("?")[0]:
        return None   # a plain TS source (tuner) of unknown codec: often MPEG-2, don't guess
    vcodec = (fmt or DEFAULT).get("vcodec", "h264")
    # Never across codecs: same codec and resolution, else the same codec's default.
    for other in [f for f in _formats().values()
                  if fmt and (f.get("width"), f.get("height")) == (fmt["width"], fmt["height"])
                  and f.get("vcodec", "h264") == vcodec] + ([DEFAULT] if vcodec == "h264" else []):
        segs = _cached(_key(dict(DEFAULT, **other)))
        if segs:
            return segs
    return None


def _cached(key):
    with _lock:
        if key in _mem:
            return _mem[key]
        path = os.path.join(CACHE_DIR, key + ".json") if CACHE_DIR else None
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    meta = json.load(f)
                with open(path[:-5] + ".ts", "rb") as f:
                    data = f.read()
                segs, pos = [], 0
                for dur, size, pts in meta:
                    segs.append((dur, data[pos:pos + size], pts))
                    pos += size
                _mem[key] = segs
                return segs
            except (OSError, ValueError):
                pass
    return None


# Common broadcast (ATSC, e.g. HDHomeRun) formats, rendered at startup so a
# tuner's first tune finds its card ready.
ATSC = [dict(DEFAULT, width=w, height=h, fps=f, channels=c, profile="Main", vcodec="mpeg2video", acodec="ac3")
        for w, h, f, c in ((1920, 1080, 30, 6), (1280, 720, 60, 6), (720, 480, 30, 2))]


def prepare():
    """Render the 1080p card, common broadcast ones and those of all source
    formats seen before (startup)."""
    segments(DEFAULT)
    for fmt in ATSC:
        segments(fmt)
    for fmt in {json.dumps(f, sort_keys=True) for f in _formats().values()}:
        segments(dict(DEFAULT, **json.loads(fmt)))


def segments(fmt):
    """[(duration, ts_bytes, start_pts)] of the slate in this format, rendering it if needed."""
    key = _key(fmt)
    segs = _cached(key)
    if segs is not None:
        return segs
    with _lock:
        if key in _rendering:
            return None
        _rendering.add(key)
    try:
        # Rendering takes seconds: outside _lock, so get() keeps answering.
        segs = render(fmt)
    finally:
        with _lock:
            _rendering.discard(key)
    log.info("rendered slate %s (%d segments)", key, len(segs))
    path = os.path.join(CACHE_DIR, key + ".json") if CACHE_DIR else None
    with _lock:
        _mem[key] = segs
        if path:
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(path[:-5] + ".ts", "wb") as f:
                    f.write(b"".join(s for _, s, _ in segs))
                with open(path, "w") as f:
                    json.dump([(d, len(s), p) for d, s, p in segs], f)
            except OSError as e:
                log.warning("can't cache slate: %s", e)
        return segs


def render(fmt):
    w, h, fps = fmt["width"] // 2 * 2, fmt["height"] // 2 * 2, int(round(fmt["fps"])) or 25
    img = Image.new("RGB", (w, h), (18, 20, 28))
    d = ImageDraw.Draw(img)
    big, small = burn.font(max(16, h * 64 // 1080)), burn.font(max(12, h * 32 // 1080))
    for text, f, y, colour in (("captions-dvr", big, 0.42, (235, 235, 235)),
                               ("captions starting...", small, 0.56, (150, 160, 175))):
        d.text(((w - f.getlength(text)) / 2, h * y), text, font=f, fill=colour)
    frame_img = np.asarray(img)

    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="mpegts")
    if fmt.get("vcodec") == "hevc":   # e.g. ATSC 3.0
        v = out.add_stream("libx265", rate=fps)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        v.options = {"preset": "veryfast", "x265-params": f"keyint={int(fps * SEG)}:min-keyint={int(fps * SEG)}:scenecut=0:log-level=error"}
    elif fmt.get("vcodec") == "mpeg2video":   # broadcast (e.g. HDHomeRun): match it
        v = out.add_stream("mpeg2video", rate=fps)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        v.options = {"g": str(int(fps * SEG)), "b": "1500k"}
    else:
        v = out.add_stream("libx264", rate=fps)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        profile = (fmt.get("profile") or "High").lower()
        profile = "baseline" if "baseline" in profile else "main" if profile == "main" else "high"
        v.options = {"preset": "veryfast", "g": str(int(fps * SEG)), "keyint_min": str(int(fps * SEG)),
                     "sc_threshold": "0", "profile": profile, "level": "4.0"}
    layout = "stereo" if fmt["channels"] >= 2 else "mono"
    acodec = "ac3" if fmt.get("acodec") == "ac3" else "aac"
    a = out.add_stream(acodec, rate=fmt["sample_rate"], layout=layout)
    fsize = a.codec_context.frame_size or 1024
    silence = np.zeros((2 if layout == "stereo" else 1, fsize), dtype=np.float32)
    samples = 0
    for i in range(int(SLATE_SECONDS * fps)):
        vf = av.VideoFrame.from_ndarray(frame_img, format="rgb24").reformat(format="yuv420p")
        vf.pts = i
        for p in v.encode(vf):
            out.mux(p)
        while samples < (i + 1) * fmt["sample_rate"] / fps:
            af = av.AudioFrame.from_ndarray(silence, format="fltp", layout=layout)
            af.sample_rate, af.pts = fmt["sample_rate"], samples
            for p in a.encode(af):
                out.mux(p)
            samples += fsize
    for s in (v, a):
        for p in s.encode(None):
            out.mux(p)
    out.close()
    return split(buf.getvalue())


def split(data):
    """Cut a TS at video keyframes (as hls.Segmenter does): [(duration, bytes, start_pts)]."""
    cuts, pts_at = [], []
    for i in range(0, len(data) - 187, 188):
        pkt = data[i:i + 188]
        if (pkt[0] == 0x47 and pkt[1] & 0x40 and ((pkt[1] & 0x1F) << 8 | pkt[2]) == hls.VIDEO_PID
                and pkt[3] & 0x20 and pkt[4] and pkt[5] & 0x40):
            pts = hls.pes_pts(pkt)
            if pts is not None:
                cuts.append(i)
                pts_at.append(pts)
    segs = []
    for k, start in enumerate(cuts):
        end = cuts[k + 1] if k + 1 < len(cuts) else len(data)
        dur = (pts_at[k + 1] - pts_at[k]) / 90000 if k + 1 < len(cuts) else SEG
        segs.append((round(dur, 3), data[start:end], pts_at[k]))
    # PAT/PMT before the first keyframe belong to the first segment.
    if cuts and segs:
        segs[0] = (segs[0][0], data[:cuts[0]] + segs[0][1], segs[0][2])
    return segs
