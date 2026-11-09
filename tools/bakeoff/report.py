"""Summarize a bake-off run: coverage, speed and a side-by-side per chunk.

Usage: python report.py out_dir [chunk_index ...]
Coverage: share of Korean speech time (large-v3 transcript segments with
Hangul) overlapped by captions that pass the relay's guard.
"""
import json
import os
import re
import sys

sys.path.insert(0, "/app")
import asr  # noqa: E402

d = sys.argv[1]
offsets = json.load(open(os.path.join(d, "offsets.json")))
systems = {n: json.load(open(os.path.join(d, n + ".json")))
           for n in ("ko", "med", "A", "B", "C") if os.path.exists(os.path.join(d, n + ".json"))}


def merge(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def total(spans):
    return sum(b - a for a, b in spans)


def inter(x, y):
    t = 0.0
    for a, b in x:
        for c, e in y:
            t += max(0.0, min(b, e) - max(a, c))
    return t


ref = merge([(o + s["start"], o + s["end"]) for o, ch in zip(offsets, systems["ko"]["chunks"])
             for s in ch["segments"] if re.search("[가-힣]", s["text"])])
print(f"reference speech: {total(ref):.0f}s of {systems['ko']['audio']:.0f}s audio\n")

kept_text = {}
for name, data in systems.items():
    if name == "ko":
        continue
    guard, spans, kept = asr.Guard(), [], []
    for o, ch, ko in zip(offsets, data["chunks"], systems["ko"]["chunks"]):
        ratio = ch.get("speech_ratio", ko.get("speech_ratio"))
        k = []
        for s in ch["segments"]:
            if not guard.keep(s, ratio):
                spans.append((o + s["start"], o + s["end"]))
                k.append(s["text"].strip())
        kept.append(" ".join(k))
    kept_text[name] = kept
    print(f"{name:4} coverage {inter(merge(spans), ref) / total(ref):5.0%}   "
          f"RTF {data['seconds'] / data['audio']:.2f}   captions {sum(1 for k in kept if k)} chunks")

for i in [int(x) for x in sys.argv[2:]] or range(len(offsets)):
    print(f"\n--- chunk {i} @ {offsets[i]:.0f}s")
    print("  KO  ", " ".join(s["text"] for s in systems["ko"]["chunks"][i]["segments"]))
    for name, kept in kept_text.items():
        print(f"  {name:4}", kept[i] or "(nothing)")
