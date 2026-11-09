"""Caption coverage of speech, old vs new ASR policy, on a 16 kHz mono WAV.

Usage: python coverage.py file.wav
Reference speech = Whisper Korean transcription over the whole file (30 s
windows, no VAD, internal filters off), segments containing Hangul.
Coverage = share of reference speech time overlapped by kept captions.
"""
import re
import sys
import time
import wave

import numpy as np

sys.path.insert(0, "/app")
import asr  # noqa: E402
from relay import find_cut  # noqa: E402

with wave.open(sys.argv[1]) as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
be = asr.LocalWhisper("medium", 12, 1, 1)
model = be._get()

segs, _ = model.transcribe(audio, language="ko", task="transcribe", beam_size=1, temperature=0.0,
                           vad_filter=False, no_speech_threshold=None, log_prob_threshold=None,
                           condition_on_previous_text=False)
ref = [(s.start, s.end) for s in segs if re.search("[가-힣]", s.text)]


def old_decode(chunk):
    out, _ = model.transcribe(chunk, language="ko", task="translate", beam_size=1,
                              temperature=0.0, vad_filter=True, condition_on_previous_text=False)
    return {"segments": [{"start": s.start, "end": s.end, "text": s.text,
                          "no_speech_prob": s.no_speech_prob, "avg_logprob": s.avg_logprob,
                          "compression_ratio": s.compression_ratio} for s in out],
            "speech_ratio": None}


chunks, pos = [], 0
while len(audio) - pos >= asr.SR * 6:
    cut = find_cut(audio[pos:pos + asr.SR * 12]) or min(len(audio) - pos, asr.SR * 12)
    chunks.append((pos / asr.SR, audio[pos:pos + cut]))
    pos += cut


def merge(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap(spans, ref):
    spans, ref, total = merge(spans), merge(ref), 0.0
    for a, b in ref:
        for c, d in spans:
            total += max(0.0, min(b, d) - max(a, c))
    return total


refsec = sum(b - a for a, b in merge(ref))
for name, fn in (("before", old_decode), ("after", be.transcribe)):
    guard, spans, music, drops, t0 = asr.Guard(), [], 0.0, {}, time.monotonic()
    for off, chunk in chunks:
        if asr.gate(chunk):
            drops["gate"] = drops.get("gate", 0) + 1
            continue
        r = fn(chunk)
        kept = 0
        for s in r["segments"]:
            why = guard.keep(s, r["speech_ratio"])
            if why:
                drops[why] = drops.get(why, 0) + 1
            else:
                kept += 1
                spans.append((off + s["start"], off + s["end"]))
        if (name == "after" and not kept and (r["speech_ratio"] is None or r["speech_ratio"] < 0.5)
                and asr.loud(chunk)):
            music += len(chunk) / asr.SR
    el = time.monotonic() - t0
    print(f"{name}: coverage {overlap(spans, ref) / refsec:.0%} of {refsec:.0f}s speech, "
          f"{len(spans)} captions, music-captioned {music:.0f}s, drops {drops}, "
          f"ASR time {el:.0f}s for {len(audio) / asr.SR:.0f}s audio")
