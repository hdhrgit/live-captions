"""The relay against synthetic sources and the fake ASR: for each format, the
start card in the source's codec, then real pictures, increasing sequence
numbers, PAT/PMT at each segment start, timed WebVTT cues, no tracebacks.
Plus the ASR going down and coming back, and a source dropping out.

Slow-ish (about a minute per run; the runs go in parallel). Set
SOURCE_FORMATS=h264_25p_aac,... to run fewer."""
import io
import json
import logging
import os
import shutil
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import av
import pytest

log = logging.getLogger("sources")

import clips
import fake_asr

RELAY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "relay")
FORMATS = [f for f in os.environ.get("SOURCE_FORMATS", ",".join(clips.FORMATS)).split(",") if f]
WATCH = 50   # seconds of relay output per run


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def asr():
    port = free_port()
    srv = fake_asr.serve(port, delay=0.1)
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except (urllib.error.URLError, OSError):
        return None


class Relay:
    def __init__(self, tmp, src_url, asr_url, name="t", label="relay"):
        self.label = label
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        cfg = {"sources": [], "settings": {}, "channels": [
            {"id": name, "name": "Test", "number": 9001, "url": src_url, "source": "manual", "enabled": True,
             "language": "en", "tvg": {}, "headers": {}}]}
        os.makedirs(tmp, exist_ok=True)
        with open(os.path.join(tmp, "channels.json"), "w") as f:
            json.dump(cfg, f)
        self.card_ms = CARD_MS.get(tmp)
        env = dict(os.environ, PORT=str(self.port), PUBLIC_URL=self.base, ASR_BACKEND="remote", ASR_URL=asr_url,
                   CHANNELS_FILE=os.path.join(tmp, "channels.json"), DELAY="12", MIN_LEAD="6", START_DELAY="4",
                   CHUNK_MIN="3", CHUNK_MAX="6", GRACE="5", ASR_WATCH="2", EPG_TRANSLATE="0")
        self.log = open(os.path.join(tmp, "relay.log"), "w+")
        self.proc = subprocess.Popen([sys.executable, "-u", "relay.py"], cwd=RELAY, env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(100):
            if get(self.base + "/status", 1):
                break
            time.sleep(0.2)
        self.name = name

    def watch(self, seconds, during=None):
        """Poll the channel like a player; returns {seq: (ts bytes, vtt text, discontinuity)}."""
        hls = f"{self.base}/hls/{self.name}"
        get(hls + "/master.m3u8", 30)
        segs, t0 = {}, time.monotonic()
        self.first_segment_after = None
        while time.monotonic() - t0 < seconds:
            if during:
                during(time.monotonic() - t0)
            pl = (get(hls + "/video.m3u8", 15) or b"").decode()
            disc = False
            for line in pl.splitlines():
                if line == "#EXT-X-DISCONTINUITY":
                    disc = True
                m = re.search(r"/(\d+)\.ts$", line)
                if m:
                    n = int(m.group(1))
                    if self.first_segment_after is None:
                        self.first_segment_after = round(time.monotonic() - t0, 1)
                    if n not in segs:
                        ts = get(f"{hls}/{n}.ts")
                        if ts:
                            segs[n] = [ts, None, disc]
                    disc = False
            for n in list(segs)[-8:]:
                segs[n][1] = (get(f"{hls}/{n}.vtt") or b"").decode()   # cues arrive later
            if time.monotonic() - getattr(self, "_logged", 0) >= 10:
                self._logged = time.monotonic()
                cues = sum((v[1] or "").count("Chunk") for v in segs.values())
                log.info("%-22s %3.0f s: %d segments, %d cues, first segment after %s s", self.label,
                         time.monotonic() - t0, len(segs), cues, self.first_segment_after)
            time.sleep(1)
        return segs

    def logtext(self):
        self.log.flush()
        self.log.seek(0)
        return self.log.read()

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def card_format(name):
    """The format the relay records for a source (Session.input_loop)."""
    vcodec, w, h, rate, _, acodec, layout = clips.FORMATS[name]
    return {"width": w, "height": h, "fps": round(float(rate)), "sample_rate": 48000,
            "channels": 6 if layout == "5.1" else 2, "profile": "Main" if vcodec != "libx264" else "High",
            "vcodec": {"libx264": "h264", "libx265": "hevc", "mpeg2video": "mpeg2video"}[vcodec],
            "acodec": "ac3" if acodec == "ac3" else "aac"}


_slate_lock = threading.Lock()


def prerender_card(cache, channel_id, fmt):
    """The start card a later tune finds. (slate.CACHE_DIR is a module global:
    one at a time, or parallel runs write each other's cards.)"""
    import slate
    with _slate_lock:
        slate.CACHE_DIR = cache
        slate._mem.clear()
        t0 = time.monotonic()
        segs = slate.segments(dict(slate.DEFAULT, **fmt))
        slate.remember(channel_id, fmt)
        ms = round((time.monotonic() - t0) * 1000)
        log.info("card %s %dx%d %s fps: %d segments in %d ms", fmt["vcodec"], fmt["width"], fmt["height"],
                 fmt["fps"], len(segs), ms)
        return ms


def first_video(ts):
    c = av.open(io.BytesIO(ts))
    v = c.streams.video[0]
    return v.codec_context.name, v.codec_context.width, v.codec_context.height


def pids(ts, n=2):
    return [((ts[i + 1] & 0x1F) << 8) | ts[i + 2] for i in range(0, 188 * n, 188)]


def check(name, segs, log):
    """Problems found in one run (empty when it's fine)."""
    vcodec = {"libx264": "h264", "libx265": "hevc", "mpeg2video": "mpeg2video"}[clips.FORMATS[name][0]]
    w, h = clips.FORMATS[name][1], clips.FORMATS[name][2]
    bad = []
    seqs = sorted(segs)
    if len(seqs) < 8:
        return [f"only {len(seqs)} segments"]
    if seqs != list(range(seqs[0], seqs[0] + len(seqs))):
        bad.append(f"sequence gaps: {seqs}")
    formats = [first_video(segs[n][0]) for n in seqs]
    if any(f[0] != vcodec for f in formats):
        bad.append(f"codec not {vcodec} throughout: {sorted(set(formats))}")
    sizes = [len(segs[n][0]) for n in seqs]
    switch = next((i for i, n in enumerate(seqs) if segs[n][2] and i > 0), None)
    if switch is None:
        bad.append("no start card (no card -> picture discontinuity)")
    else:
        if not all(f[1:] == (w, h) for f in formats):
            bad.append(f"card or picture not {w}x{h}: {sorted(set(formats))}")
        if sizes[switch - 1] * 1.2 > sizes[switch]:
            bad.append(f"no bigger real picture after the card: {sizes[switch - 3:switch + 2]}")
    for n in seqs[switch or 0:]:
        if pids(segs[n][0]) != [0, 0x1000]:
            bad.append(f"segment {n} doesn't start with PAT, PMT: {pids(segs[n][0])}")
            break
    vtts = [segs[n][1] or "" for n in seqs]
    if not all(v.startswith("WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:0,LOCAL:00:00:00.000") for v in vtts if v):
        bad.append("VTT without the timestamp map")
    if switch and not any("starting..." in v for v in vtts[:switch]):
        bad.append("no status cue (\"... starting...\") over the start card")
    cues = [v for v in vtts if "Chunk" in v]
    if not cues:
        bad.append("no cues from the fake ASR")
    if "Traceback" in log:
        bad.append("traceback in the log: " + log[log.index("Traceback"):][:600])
    return bad


def run_format(name, asr, tmp, kind="hls", cold=False):
    """cold: a new install's first tune of this source (no card cached for
    its format); else the card a later tune finds."""
    src = clips.Source(name).start()
    test = f"{kind}-{'cold' if cold else 'warm'}"
    log.info("%-22s start (%s source)", f"{test} {name}", kind)
    relay = Relay(run_dir(tmp, name, test), src.hls_url if kind == "hls" else src.ts_url, asr,
                  label=f"{test} {name}")
    try:
        segs = relay.watch(WATCH + (15 if kind == "ts" else 0))
        problems = check(name, segs, relay.logtext())
        first = relay.first_segment_after
        if first is None or first > 5:
            problems.append(f"first segment after {first} s, expected under 5 s")
        stats = run_stats(name, segs, relay, first)
        if not (stats["lead_s"] or 0) > 0:
            problems.append(f"lead {stats['lead_s']} s at the end, expected > 0 (output fell behind the source)")
        stats["card_ms"] = relay.card_ms
        report(test, name, stats, problems)
        keep(test, name, segs, relay)
        return name, problems
    finally:
        relay.stop()
        src.close()


REPORT = os.environ.get("TEST_REPORT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report"))
_report_lock = threading.Lock()


def keep(test, name, segs, relay):
    """The relay's log, the first real segment and a VTT with cues, for the CI artifact."""
    d = os.path.join(REPORT, test, name)
    os.makedirs(d, exist_ok=True)
    shutil.copy(relay.log.name, os.path.join(d, "relay.log"))
    seqs = sorted(segs)
    real = next((n for n in seqs if segs[n][2]), seqs[0] if seqs else None)
    if real is not None:
        open(os.path.join(d, f"first-segment-{real}.ts"), "wb").write(segs[real][0])
    vtt = next((segs[n][1] for n in seqs if "Chunk" in (segs[n][1] or "")), None)
    if vtt:
        open(os.path.join(d, "sample.vtt"), "w").write(vtt)


CARD_MS = {}   # run directory -> card render time (warm runs)


def run_dir(tmp, name, test):
    return os.path.join(tmp, f"{name}-{test}")


def prerender_all(tmp, names, kind):
    """Warm runs: the cards a later tune finds, rendered one by one before any
    source starts (rendering them under the sources' clocks made the runs late)."""
    for n in names:
        d = run_dir(tmp, n, f"{kind}-warm")
        CARD_MS[d] = prerender_card(os.path.join(d, "slate"), "t", card_format(n))


def run_stats(name, segs, relay, first):
    """What a run produced, for the report."""
    seqs = sorted(segs)
    switch = next((i for i, n in enumerate(seqs) if segs[n][2] and i > 0), None)
    cues = set()
    for n in seqs:
        cues.update(re.findall(r"Chunk \d+ part \d+", segs[n][1] or ""))
    st = json.loads(get(relay.base + "/status") or b"{}").get("channels", {}).get(relay.name, {})
    sess = st.get("session") or {}
    real = [len(segs[n][0]) for n in seqs[switch or 0:]]
    return {"segments": len(seqs), "first_segment_s": first, "card_segments": switch or 0,
            "avg_kb": round(sum(real) / len(real) / 1024) if real else 0, "cues": len(cues),
            "captions": sess.get("captions"), "dropped": sess.get("dropped") or {},
            "lead_s": sess.get("lead_s")}


def report(test, name, stats, problems):
    """One log line per run, and tests/report/<test>.json (the summary table)."""
    line = (f"{test + ' ' + name:<22} done: {stats['segments']} segments ({stats['card_segments']} card), "
            f"first after {stats['first_segment_s']} s, {stats['avg_kb']} KB/segment, {stats['cues']} cues, "
            f"{stats['captions']} captions, lead {stats['lead_s']} s, dropped {stats['dropped'] or '-'}: "
            f"{'OK' if not problems else 'FAIL: ' + '; '.join(problems)}")
    with _report_lock:
        (log.info if not problems else log.error)(line)
        os.makedirs(REPORT, exist_ok=True)
        path = os.path.join(REPORT, f"{test}.json")
        data = json.load(open(path)) if os.path.exists(path) else {}
        data[name] = dict(stats, problems=problems)
        json.dump(data, open(path, "w"), indent=1)


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_formats_hls(asr, tmp_path, cold):
    if not cold:
        prerender_all(str(tmp_path), FORMATS, "hls")
    # Cold runs render their cards while they start: three at a time, or the
    # renders compete for the CPU more than one tune at a time ever would.
    with ThreadPoolExecutor(3 if cold else len(FORMATS)) as ex:
        results = list(ex.map(lambda n: run_format(n, asr, str(tmp_path), cold=cold), FORMATS))
    problems = {n: p for n, p in results if p}
    assert not problems, json.dumps(problems, indent=1)


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_plain_ts_sources(asr, tmp_path, cold):
    """A plain MPEG-TS source (an HDHomeRun tuner): MPEG-2 1080i and H.264."""
    names = [n for n in ("mpeg2_2997i_ac3", "h264_2997p_aac") if n in FORMATS]
    if not cold:
        prerender_all(str(tmp_path), names, "ts")
    with ThreadPoolExecutor(len(names)) as ex:
        results = list(ex.map(lambda n: run_format(n, asr, str(tmp_path), "ts", cold), names))
    problems = {n: p for n, p in results if p}
    assert not problems, json.dumps(problems, indent=1)


def set_mode(asr, mode):
    """Switch the fake ASR (ok, down, error, slow); logged when it changes."""
    if fake_asr.STATE["mode"] != mode:
        log.info("fake ASR -> %s", mode)
    urllib.request.urlopen(urllib.request.Request(asr + f"/mode?set={mode}", method="POST", data=b""))


def test_asr_down_and_back(asr, tmp_path):
    src = clips.Source("h264_25p_aac").start()
    relay = Relay(str(tmp_path / "asrdown"), src.hls_url, asr)
    try:
        def toggle(t):
            if 20 < t < 21:
                set_mode(asr, "down")
            if 32 < t < 33:
                set_mode(asr, "ok")
        segs = relay.watch(50, toggle)
        log_text = relay.logtext()
        st = json.loads(get(relay.base + "/status"))
    finally:
        set_mode(asr, "ok")
        relay.stop()
        src.close()
    for line in log_text.splitlines():
        if "speech server" in line:
            log.info("asr-down relay: %s", line.split(" ", 3)[-1][:110])
    log.info("asr-down: %d segments while the ASR was down for 12 s", len(segs))
    assert log_text.count("speech server unreachable") == 1, log_text[-2000:]
    assert log_text.count("speech server") >= 2   # up, down, up
    assert "Traceback" not in log_text
    assert sorted(segs) == list(range(min(segs), max(segs) + 1)), "video kept going"
    assert st["asr_health"]["ok"]


def test_source_dropout(asr, tmp_path):
    src = clips.Source("h264_25p_aac").start()
    relay = Relay(str(tmp_path / "dropout"), src.hls_url, asr)
    try:
        segs = relay.watch(60, lambda t: src.stop_for(12) if 20 < t < 21 else None)
        log_text = relay.logtext()
    finally:
        relay.stop()
        src.close()
    seqs = sorted(segs)
    log.info("dropout: %d segments %d..%d, input errors logged: %d, discontinuities: %d", len(seqs), seqs[0],
             seqs[-1], log_text.count("input error"), sum(1 for n in seqs if segs[n][2]))
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs))), "no sequence jumps"
    assert "Traceback" not in log_text
    assert "input error" in log_text   # it noticed, and carried on
    assert seqs[-1] - seqs[0] > 15   # it came back (a 12 s outage + 5 s retry outruns the 12 s lead)
