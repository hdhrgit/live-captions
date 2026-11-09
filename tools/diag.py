"""Per-chunk diagnosis of where speech goes missing.

Usage: python diag.py file.wav [max_chunks]
For each relay-style chunk prints: span, pre-gate, Silero VAD speech ratio,
segments from the relay's current settings, segments with Whisper's internal
filters off (with metrics and guard verdict), and a Korean transcript.
"""
import sys
import wave

import numpy as np

sys.path.insert(0, "/app")
import asr  # noqa: E402
from relay import find_cut  # noqa: E402
from faster_whisper.vad import VadOptions, get_speech_timestamps  # noqa: E402

with wave.open(sys.argv[1]) as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10**6

model = asr.LocalWhisper("medium", 12, 1, 1)._get()
guard = asr.Guard()


def run(chunk, **kw):
    opts = dict(language="ko", task="translate", beam_size=1, temperature=0.0,
                condition_on_previous_text=False)
    opts.update(kw)
    segs, _ = model.transcribe(chunk, **opts)
    return list(segs)


pos, n = 0, 0
while len(audio) - pos >= asr.SR * 6 and n < limit:
    cut = find_cut(audio[pos:pos + asr.SR * 12]) or min(len(audio) - pos, asr.SR * 12)
    chunk = audio[pos:pos + cut]
    vad = get_speech_timestamps(chunk, VadOptions())
    ratio = sum(v["end"] - v["start"] for v in vad) / len(chunk)
    cur = run(chunk, vad_filter=True)
    raw = run(chunk, vad_filter=False, no_speech_threshold=None, log_prob_threshold=None)
    ko = run(chunk, task="transcribe", vad_filter=False, no_speech_threshold=None,
             log_prob_threshold=None)
    print(f"=== {pos / asr.SR:6.1f}-{(pos + cut) / asr.SR:6.1f}s gate={asr.gate(chunk)} "
          f"vad={ratio:.0%} current={len(cur)} seg")
    for s in raw:
        d = {"text": s.text, "no_speech_prob": s.no_speech_prob,
             "avg_logprob": s.avg_logprob, "compression_ratio": s.compression_ratio}
        print(f"  EN nsp={s.no_speech_prob:.2f} lp={s.avg_logprob:.2f} "
              f"cr={s.compression_ratio:.2f} guard={guard.keep(d) or 'keep'}: {s.text.strip()}")
    print("  KO", " ".join(s.text.strip() for s in ko))
    pos += cut
    n += 1
