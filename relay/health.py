"""Stream health: probe channel URLs directly and remember what works.

For each URL: fetch the playlist (with the source's referrer/user-agent),
pick the variant the relay would use, then read a few seconds of media with
PyAV. The result is stored on the channel as "health" and drives the picker's
✅/❌, the "hide broken" filter and URL failover between feeds of one channel.
"""
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import av

log = logging.getLogger("health")
HOURS = float(os.environ.get("HEALTH_HOURS", "12"))
WORKERS = int(os.environ.get("HEALTH_WORKERS", "6"))
SECONDS = 6.0


def probe(url, headers, pick_variant):
    """Probe one URL; returns a health dict."""
    t0 = time.monotonic()
    res = {"ok": False, "checked": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    h = {"User-Agent": headers.get("user_agent") or "Mozilla/5.0"}
    if headers.get("referer"):
        h["Referer"] = headers["referer"]
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=10) as r:
            r.read(1024)
    except urllib.error.HTTPError as e:
        res["error"] = f"HTTP {e.code}" + (" (forbidden or geo-blocked)" if e.code in (401, 403, 451) else "")
        return res
    except Exception as e:  # noqa: BLE001
        res["error"] = "timeout" if "timed out" in str(e) else f"{type(e).__name__}: {str(e)[:80]}"
        return res
    target = pick_variant(url, headers)
    res = media(target, headers, h, res, t0)
    if not res["ok"] and target != url:
        res = media(url, headers, h, {k: v for k, v in res.items() if k in ("checked",)}, t0)
    return res


def media(target, headers, h, res, t0):
    """Read a few seconds of audio/video from target into res."""
    res = dict(res, ok=False)
    options = {"user_agent": h["User-Agent"]}
    if headers.get("referer"):
        options["headers"] = f"Referer: {headers['referer']}\r\n"
    try:
        with av.open(target, timeout=(10, 10), options=options) as c:
            v = c.streams.video[0] if c.streams.video else None
            a = c.streams.audio[0] if c.streams.audio else None
            res["video"] = v.codec_context.name if v else None
            res["audio"] = a.codec_context.name if a else None
            res["height"] = v.codec_context.height if v else None
            first = None
            for pkt in c.demux(*(s for s in (v, a) if s)):
                if pkt.dts is None or pkt.time_base is None:
                    continue
                t = float(pkt.dts * pkt.time_base)
                first = t if first is None else first
                if t - first >= SECONDS or time.monotonic() - t0 > 25:
                    break
            got = 0 if first is None else t - first
        if not v or not a:
            res["error"] = "no video" if not v else "no audio"
        elif got < 2:
            res["error"] = f"stalled ({got:.0f}s of media)"
        else:
            res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:100]}"
    res["seconds"] = round(time.monotonic() - t0, 1)
    return res


class Checker:
    def __init__(self, cfg, pick_variant, config):
        self.cfg, self.pick_variant, self.config = cfg, pick_variant, config
        self.running = False
        self.progress = None
        self.wake = threading.Event()

    def targets(self):
        """Major channels in added sources, plus every enabled channel."""
        with self.config.lock:
            return [c for c in self.cfg.channels if c.get("major") or c.get("enabled")]

    def run(self):
        chans = self.targets()
        self.running, self.progress = True, [0, len(chans)]
        log.info("checking %d streams", len(chans))

        def one(c):
            r = probe(c["url"], c.get("headers", {}), self.pick_variant)
            with self.config.lock:
                c["health"] = r
            self.progress[0] += 1
            return r["ok"]

        with ThreadPoolExecutor(WORKERS) as pool:
            ok = sum(pool.map(one, chans))
        self.cfg.save()
        self.running = False
        log.info("stream check done: %d of %d ok", ok, len(chans))

    def loop(self):
        while True:
            self.wake.wait(HOURS * 3600)
            self.wake.clear()
            try:
                self.run()
            except Exception as e:  # noqa: BLE001
                self.running = False
                log.warning("stream check failed: %s", e)

    def trigger(self):
        if not self.running:
            self.wake.set()
