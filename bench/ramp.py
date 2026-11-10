#!/usr/bin/env python3
"""Load test: ramp up streams on a relay and print a markdown table.

Tunes test streams through the relay's stateless /api/stream (so the
relay's configured channels stay untouched), keeps each one "watched" by
polling its playlist, and after each step measures for a while:
ASR time per chunk (median, p90), chunks skipped vs captions made, GPU use
(from the ASR server's /metrics, if it has a GPU) and relay CPU (from the
relay's /metrics).

  bench/ramp.py --relay http://<relay-host>:8095 --asr http://<asr-host>:8096 \\
      --group english --mt whisper --lang en --steps 4,8,12,16

Streams already running on the relay (people watching) are counted too.
Stop with Ctrl-C; the test streams stop on their own about two minutes
after the script stops polling them.
"""
import argparse
import json
import random
import re
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request

HERE = __file__.rsplit("/", 1)[0]


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def alive(s):
    """The source answers with a playlist or TS data."""
    try:
        req = urllib.request.Request(s["url"], headers={"User-Agent": s.get("user_agent") or "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return bool(r.read(2048))
    except Exception:  # noqa: BLE001 - dead or blocked: skip it
        return False


def metrics(url):
    """{name{labels}: value} from a Prometheus text endpoint, or {}."""
    out = {}
    try:
        for line in get(url + "/metrics", 5).decode().splitlines():
            if line and not line.startswith("#"):
                k, _, v = line.rpartition(" ")
                out[k] = float(v)
    except Exception:  # noqa: BLE001 - no metrics there
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relay", required=True)
    ap.add_argument("--asr", help="ASR server, for GPU use")
    ap.add_argument("--group", default="english", help="streams.json group: english or foreign")
    ap.add_argument("--mt", default="whisper", help="seamless, whisper or nllb")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--chunk", help="long: 20-28 s chunks")
    ap.add_argument("--steps", default="4,8,12,16", help="test streams to have running at each step")
    ap.add_argument("--settle", type=float, default=75, help="seconds after adding streams before measuring")
    ap.add_argument("--measure", type=float, default=100, help="seconds measured per step")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    relay = a.relay.rstrip("/")

    pool = json.load(open(f"{HERE}/streams.json"))[a.group]
    random.Random(a.seed).shuffle(pool)
    steps = [int(x) for x in a.steps.split(",")]
    print(f"checking {len(pool)} {a.group} sources...", file=sys.stderr)
    ok = []
    for s in pool:
        if len(ok) >= max(steps) + 4:   # a few spare for ones the relay refuses
            break
        if alive(s):
            ok.append(s)
    if len(ok) < max(steps):
        print(f"only {len(ok)} sources answer", file=sys.stderr)

    active, stop = [], threading.Event()

    def keep():   # poll like a player, so the relay keeps the streams running
        while not stop.is_set():
            for base in list(active):
                try:
                    get(base + "/video.m3u8", 5)
                except Exception:  # noqa: BLE001
                    pass
            stop.wait(4)
    threading.Thread(target=keep, daemon=True).start()

    def tune(s):
        q = {"url": s["url"], "mt": a.mt, "lang": a.lang, "name": s["name"]}
        if a.chunk:
            q["chunk"] = a.chunk
        if s.get("user_agent"):
            q["user_agent"] = s["user_agent"]
        body = get(f"{relay}/api/stream?{urllib.parse.urlencode(q)}", 30).decode()
        m = re.search(r"(https?://[^\s]+/hls/[^/\s]+)/video\.m3u8", body)
        return m.group(1).replace(m.group(1).split("/hls/")[0], relay) if m else None

    def sessions(adhoc_only=False):
        st = json.loads(get(relay + "/status"))
        chans = list(st["adhoc"].values()) + ([] if adhoc_only else list(st["channels"].values()))
        return [c["session"] for c in chans if c.get("session")]

    def totals():
        made = skipped = 0
        for s in sessions():
            made += s.get("captions", 0)
            skipped += sum((s.get("dropped") or {}).get(r, 0) for r in ("asr_down", "asr_busy"))
        return made, skipped

    def cpu():
        m = metrics(relay)
        return sum(v for k, v in m.items() if k.endswith("cpu_seconds_total") or "_cpu_seconds_total{" in k)

    print(f"| Streams | ASR median / p90 | Skipped / made | GPU | Relay CPU |  ({a.group}, mt={a.mt}"
          f"{', chunk=' + a.chunk if a.chunk else ''})")
    print("|---|---|---|---|---|")
    spare = iter(ok)
    try:
        for n in steps:
            while len(active) < n:
                s = next(spare, None)
                if s is None:
                    break
                try:
                    base = tune(s)
                    if base:
                        active.append(base)
                except Exception as e:  # noqa: BLE001 - refused or dead: next one
                    print(f"  skip {s['name']}: {e}", file=sys.stderr)
            time.sleep(a.settle)
            made0, skipped0 = totals()
            cpu0, t0 = cpu(), time.time()
            times, gpu = [], []
            while time.time() - t0 < a.measure:
                # (the test streams only: a paused channel keeps its last value)
                times += [s["asr_last_s"] for s in sessions(adhoc_only=True) if s.get("asr_last_s") is not None]
                if a.asr:
                    g = metrics(a.asr.rstrip("/")).get("asr_gpu_utilization_percent")
                    if g is not None:
                        gpu.append(g)
                time.sleep(10)
            made1, skipped1 = totals()
            cpu_pct = (cpu() - cpu0) / (time.time() - t0) * 100
            times.sort()
            med = statistics.median(times) if times else float("nan")
            p90 = times[int(len(times) * 0.9)] if times else float("nan")
            print(f"| {len(sessions())} | {med:.1f} s / {p90:.1f} s | {skipped1 - skipped0} / {made1 - made0} | "
                  f"{statistics.mean(gpu):.0f}% | {cpu_pct:.0f}% |" if gpu else
                  f"| {len(sessions())} | {med:.1f} s / {p90:.1f} s | {skipped1 - skipped0} / {made1 - made0} | - | "
                  f"{cpu_pct:.0f}% |", flush=True)
    finally:
        stop.set()


if __name__ == "__main__":
    main()
