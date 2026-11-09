"""Bake-off: which Korean->English pipeline captions KCTV best.

Usage: python bakeoff.py file.wav out_dir [mac_url]
Chunks the audio like the relay, then per chunk runs:
  ko      Whisper large-v3 transcribe (Korean reference, and input to B)
  med     Whisper medium translate (CPU, the current fallback)
  A       Whisper large-v3 translate (on the Mac via the HTTP contract if
          mac_url is given, else on this CPU)
  B       ko segments -> NLLB-200 distilled 1.3B
  C       SeamlessM4T v2 large speech -> English text
Each system's results are cached in out_dir/<system>.json so a run can resume.
"""
import json
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, "/app")
import asr  # noqa: E402
from relay import find_cut  # noqa: E402

THREADS = int(os.environ.get("THREADS", "6"))
wav_path, out_dir = sys.argv[1], sys.argv[2]
mac = sys.argv[3] if len(sys.argv) > 3 else None
os.makedirs(out_dir, exist_ok=True)

with wave.open(wav_path) as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
chunks, pos = [], 0
while len(audio) - pos >= asr.SR * 3:
    cut = find_cut(audio[pos:pos + asr.SR * 12]) or min(len(audio) - pos, asr.SR * 12)
    chunks.append((pos / asr.SR, audio[pos:pos + cut]))
    pos += cut
print(f"{len(chunks)} chunks", flush=True)
json.dump([p for p, _ in chunks], open(os.path.join(out_dir, "offsets.json"), "w"))


def cached(name, fn):
    path = os.path.join(out_dir, name + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.monotonic()
    res = [fn(c) for _, c in chunks]
    el = time.monotonic() - t0
    data = {"seconds": el, "audio": len(audio) / asr.SR, "chunks": res}
    json.dump(data, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"{name}: {el:.0f}s for {len(audio) / asr.SR:.0f}s audio (RTF {el / (len(audio) / asr.SR):.2f})",
          flush=True)
    return data


def segs(it):
    return [{"start": s.start, "end": s.end, "text": s.text.strip(),
             "no_speech_prob": s.no_speech_prob, "avg_logprob": s.avg_logprob,
             "compression_ratio": s.compression_ratio} for s in it]


def whisper_run(model, task):
    def fn(c):
        it, _ = model.transcribe(c, language="ko", task=task, beam_size=1, temperature=[0.0, 0.4],
                                 compression_ratio_threshold=2.4, vad_filter=False,
                                 no_speech_threshold=None, log_prob_threshold=None,
                                 condition_on_previous_text=False)
        return {"segments": segs(it), "speech_ratio": asr.speech_ratio(c)}
    return fn


from faster_whisper import WhisperModel  # noqa: E402

large = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=THREADS,
                     download_root="/models")
ko = cached("ko", whisper_run(large, "transcribe"))
del large
med = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=THREADS,
                   download_root="/models")
cached("med", whisper_run(med, "translate"))
del med

if mac:
    remote = asr.Remote(mac, timeout=60)

    def mac_chunk(c):
        for attempt in range(10):  # ride out Mac restarts and naps
            try:
                t0 = time.monotonic()
                r = remote.transcribe(c)
                # Pace to <= 1/3 of the Mac so the live relay keeps priority.
                time.sleep(2 * (time.monotonic() - t0))
                return r
            except OSError as e:
                print(f"A: retry after {e}", flush=True)
                time.sleep(15)
        raise RuntimeError("Mac ASR unavailable")

    cached("A", mac_chunk)

import torch  # noqa: E402
from transformers import (AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer,  # noqa: E402
                          SeamlessM4Tv2ForSpeechToText)

torch.set_num_threads(THREADS)
proc = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
sm = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large").eval()


def seamless_chunk(c):
    with torch.inference_mode():
        inputs = proc(audio=c, sampling_rate=asr.SR, return_tensors="pt")
        ids = sm.generate(**inputs, tgt_lang="eng", max_new_tokens=256)
    text = proc.decode(ids[0].tolist(), skip_special_tokens=True)
    return {"segments": [{"start": 0.0, "end": len(c) / asr.SR, "text": text}]}


cached("C", seamless_chunk)
del sm

tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="kor_Hang")
nllb = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").eval()
ko_iter = iter(ko["chunks"])


def nllb_chunk(_c):
    out = []
    for s in next(ko_iter)["segments"]:
        with torch.inference_mode():
            ids = nllb.generate(**tok(s["text"], return_tensors="pt"), max_new_tokens=200,
                                forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"))
        out.append({**s, "ko": s["text"], "text": tok.batch_decode(ids, skip_special_tokens=True)[0]})
    return {"segments": out}


cached("B", nllb_chunk)
del nllb


if not mac:
    # A on the Docker host's CPU: the same large-v3 model the Mac runs, translate task.
    large = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=THREADS,
                         download_root="/models")
    cached("A", whisper_run(large, "translate"))
    del large

print("done", flush=True)
