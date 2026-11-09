"""Guard drop reasons per system, and side-by-side of forecast and news chunks."""
import json
import sys

sys.path.insert(0, "/app")
import asr  # noqa: E402

d = sys.argv[1]
off = json.load(open(f"{d}/offsets.json"))
S = {n: json.load(open(f"{d}/{n}.json")) for n in ("ko", "med", "A", "B", "C")}
for n in ("med", "A", "B", "C"):
    g, reasons = asr.Guard(), {}
    for ch, ko in zip(S[n]["chunks"], S["ko"]["chunks"]):
        r = ch.get("speech_ratio", ko.get("speech_ratio"))
        for s in ch["segments"]:
            why = g.keep(s, r)
            if why:
                reasons[why] = reasons.get(why, 0) + 1
    print(n, reasons)
idx = ([i for i, o in enumerate(off) if 690 <= o <= 760] + [i for i, o in enumerate(off) if 100 <= o <= 112]
       + [i for i, o in enumerate(off) if 300 <= o <= 312])
for i in idx[:9]:
    print(f"\n--- {off[i]:.0f}s")
    print("  KO ", " ".join(s["text"] for s in S["ko"]["chunks"][i]["segments"])[:300])
    for n in ("A", "B", "C"):
        print(f"  {n}  ", " ".join(s["text"].strip() for s in S[n]["chunks"][i]["segments"])[:300])
