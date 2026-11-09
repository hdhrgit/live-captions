"""Run a WAV through the relay's chunking, pre-gate, ASR and guard.

Usage: python guard_test.py file.wav   (16 kHz mono)
Prints what Whisper produced per chunk and whether the gate/guard kept it,
as the relay runs it (gated chunks are decoded too, for comparison).
"""
import sys
import wave

import numpy as np

sys.path.insert(0, "/app")
import asr  # noqa: E402
from relay import find_cut  # noqa: E402

with wave.open(sys.argv[1]) as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768

chunks, pos = [], 0
while len(audio) - pos >= asr.SR * 12:
    cut = find_cut(audio[pos:pos + asr.SR * 12]) or asr.SR * 12
    chunks.append((pos, audio[pos:pos + cut]))
    pos += cut

model = asr.LocalWhisper("medium", 6, 1, 1)
guard = asr.Guard()
raw = kept = gated = music = 0
print(f"=== {len(chunks)} chunks")
for pos, chunk in chunks:
    g = asr.gate(chunk)
    gated += bool(g)
    # Decode even gated chunks, to show what the gate saved us from.
    r = model.transcribe(chunk)
    k = 0
    for s in r["segments"]:
        raw += 1
        reason = g or guard.keep(s, r["speech_ratio"])
        k += reason is None
        print(f"[{pos / asr.SR + s['start']:6.1f}] speech={r['speech_ratio']} "
              f"nsp={s['no_speech_prob']:.2f} lp={s['avg_logprob']:.2f} "
              f"cr={s['compression_ratio']:.2f} "
              f"{'KEEP' if reason is None else 'drop:' + reason:14} {s['text'].strip()}")
    kept += k
    if not g and not k and (r["speech_ratio"] or 0) < 0.5 and asr.loud(chunk):
        music += 1
print(f"--- raw segments {raw}, kept {kept}, music captions {music}; "
      f"chunks gated before ASR: {gated}/{len(chunks)}")
