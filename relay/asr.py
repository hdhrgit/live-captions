"""ASR/translation backends and the hallucination guard.

Every backend takes 16 kHz mono float32 audio and returns
{"segments": [{start, end, text, no_speech_prob?, avg_logprob?,
compression_ratio?}], "speech_ratio": float|None}, times in seconds from the
start of the chunk. The guard is applied by
the relay, so it behaves the same whatever backend produced the segments.
"""
import collections
import io
import json
import logging
import os
import re
import threading
import time
import urllib.request
import wave
import zlib

import numpy as np

SR = 16000
# Cap per 30 s window: a 12 s chunk of speech needs ~60 tokens, and a decode
# stuck in a repetition loop would otherwise run to 448 and hold the GPU.
MAX_TOKENS = int(os.environ.get("ASR_MAX_TOKENS", "160"))
log = logging.getLogger("asr")


# ---------------------------------------------------------------- guard ---

BLOCKLIST = re.compile(r"|".join([
    r"thank(s| you) (very much )?for (watching|listening|viewing)",
    r"please (like|subscribe)", r"\bsubscribe\b", r"like and subscribe",
    r"subtitles? (by|provided)", r"captions? by", r"translated by",
    r"amara\.org", r"www\.", r"\.com\b", r"MBC|KBS|SBS", r"copyright",
    r"see you (next time|in the next)", r"don'?t forget to",
    r"^(thank you|thanks|bye|you|okay|oh|ah|um|hmm)[.!]?$",
    r"(this video|it) (includes|contains) paid", r"paid (ads|promotion|advertis)",
    r"see you in the next video", r"in this video,? (we|i)'?(ll| will)"
]), re.I)

MUSIC = re.compile(r"[♪♫♬♯♩]|^\W*(music|applause|laughter|silence)\W*$", re.I)


# Languages Whisper can recognise (ISO 639-1, as Whisper names them).
WHISPER_LANGS = set("""af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr
gl gu ha haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn
mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn so sq sr su sv sw ta te tg th
tk tl tr tt uk ur uz vi yi yo yue zh""".split())


def junk(text):
    """Reason a segment's text is not a caption whatever the audio, or None."""
    text = text.strip()
    if MUSIC.search(text):
        return "music"
    if not re.search(r"[A-Za-z]{2}", text):
        return "no words"
    if BLOCKLIST.search(text):
        return "blocklist"
    return None


class Guard:
    """Drops segments that are probably not speech.

    Confidence filters are strict only where VAD found little speech (bars,
    tone, music, noise). Loud, emotional delivery scores a high
    no_speech_prob and a low avg_logprob on real speech, so over speech
    only very low confidence is dropped.
    """

    def __init__(self, no_speech=0.6, logprob=-1.0, hard_logprob=-1.5,
                 loose_logprob=-2.0, compression=2.4, speech=0.3, repeat_window=4):
        self.no_speech = no_speech
        self.logprob = logprob
        self.hard_logprob = hard_logprob
        self.loose_logprob = loose_logprob
        self.compression = compression
        self.speech = speech
        self.recent = []
        self.repeat_window = repeat_window

    def keep(self, seg, speech_ratio=None):
        """Returns None to keep the segment, or the reason it was dropped."""
        text = seg["text"].strip()
        nsp = seg.get("no_speech_prob")
        lp = seg.get("avg_logprob")
        cr = seg.get("compression_ratio")
        reason = junk(text)
        if reason:
            return reason
        if cr is not None and cr > self.compression:
            return "compression"
        if speech_ratio is not None and speech_ratio >= self.speech:
            if lp is not None and lp < self.loose_logprob:
                return "logprob"
        else:
            if nsp is not None and lp is not None and nsp > self.no_speech and lp < self.logprob:
                return "no_speech"
            if lp is not None and lp < self.hard_logprob:
                return "logprob"
        norm = re.sub(r"\W+", " ", text.lower()).strip()
        if norm in self.recent:
            return "repeat"
        self.recent = (self.recent + [norm])[-self.repeat_window:]
        return None


REPEATED_PHRASE = re.compile(r"\b(\w+(?:\W+\w+){0,3})(?:\W+\1\b){3,}", re.I)


def collapse_repeats(text):
    """Drops sentences repeated within one cue ("I'm not exposed. - I'm not
    exposed. - I'm not exposed."), keeping the first of each."""
    out, seen = [], set()
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        norm = re.sub(r"\W+", " ", sentence.lower()).strip()
        if norm and norm in seen:
            continue
        seen.add(norm)
        out.append(sentence)
    return " ".join(out)


def seamless_check(seg, speech_ratio=None):
    """Extra checks for Seamless output, which has no confidence scores and
    produces fluent text from music and noise: reason to drop, or None.
    Collapses repeated sentences in seg["text"] first."""
    text = seg["text"] = collapse_repeats(seg["text"])
    if speech_ratio is not None and speech_ratio < 0.25:
        return "low_speech"
    if REPEATED_PHRASE.search(text):
        return "loop"
    if len(text) >= 40 and len(text) / max(1, len(zlib.compress(text.encode()))) > 2.4:
        return "compression"
    # More words than anyone says in the speech VAD found: invented.
    if speech_ratio is not None:
        speech_s = speech_ratio * (seg["end"] - seg["start"])
        if len(text.split()) > 5 * speech_s + 6:
            return "too_long"
    return None


def speech_regions(audio, min_gap=0.3, min_len=0.15):
    """(start, end) seconds of the louder, speech-like parts of a chunk, by
    frame energy (20 ms frames; gaps under min_gap merged). Used to place
    subtitle text on the speech when a backend gives no timestamps."""
    win = SR // 50
    n = len(audio) // win
    if n == 0:
        return []
    rms = np.sqrt(np.mean(audio[:n * win].reshape(n, win) ** 2, axis=1))
    rms = np.convolve(rms, np.ones(3) / 3, mode="same")
    thr = max(10 ** (-45 / 20), 0.2 * float(np.percentile(rms, 90)))
    regions, cur = [], None
    for i, on in enumerate(rms > thr):
        if on:
            if cur and i - cur[1] <= min_gap * 50:
                cur[1] = i + 1
            else:
                if cur:
                    regions.append(cur)
                cur = [i, i + 1]
    if cur:
        regions.append(cur)
    return [(s / 50, e / 50) for s, e in regions if (e - s) / 50 >= min_len]


def loud(audio, dbfs=-40):
    return len(audio) > 0 and float(np.sqrt(np.mean(audio ** 2))) > 10 ** (dbfs / 20)


def gate(audio):
    """Cheap pre-check before ASR: reason to skip the chunk, or None.

    Catches digital silence and the steady test tone that runs with bars,
    which would otherwise reach Whisper (and a remote backend) every chunk.
    """
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    if rms < 10 ** (-50 / 20):
        return "silence"
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
    total = spec.sum()
    if total > 0:
        # Energy within +-10 Hz of the strongest frequency
        peak = int(np.argmax(spec))
        bw = max(1, int(10 * len(audio) / SR))
        if spec[max(0, peak - bw):peak + bw + 1].sum() / total > 0.6:
            return "tone"
    return None


# ------------------------------------------------------------- backends ---

class Whisper:
    """Shared decode policy for Whisper engines.

    No VAD cut and no internal no-speech/logprob skipping (both drop loud
    speech); the relay's gate and Guard handle non-speech instead. Each
    window falls back to a higher temperature only when it loops
    (compression ratio > 2.4). A chunk that VAD says is mostly speech but
    decodes to nothing usable is retried once, warmer.
    """

    RETRY_SPEECH = 0.5

    def transcribe(self, audio, language="ko", task="translate", mt="whisper"):
        if mt == "seamless" and task == "translate" and seamless() is not None:
            ratio = speech_ratio(audio)
            text = seamless().translate(audio)
            seg = {"start": 0.0, "end": len(audio) / SR, "text": " " + text}
            return {"segments": [seg] if text.strip() else [], "speech_ratio": ratio,
                    "mt": "seamless", "language": language}
        return self._transcribe(audio, language, task, mt)

    def multi(self, audio, language=None, tracks=("nllb", "whisper", "original")):
        """All requested tracks for one chunk, overlapping CPU and GPU work:
        Whisper transcribes (GPU), then NLLB translates that transcript on the
        CPU in a thread while Whisper translates (GPU) in parallel.
        Returns {"tracks": {name: segments}, "speech_ratio", "language"}."""
        ratio = speech_ratio(audio)
        src_segs, info = self.decode(audio, language, "transcribe", (0.0, 0.4))
        src = language or info.get("language")
        out = {"original": [dict(x) for x in src_segs]}
        if src == "en":
            out["nllb"] = out["whisper"] = src_segs
        else:
            nllb_res = {}
            job = None
            if "nllb" in tracks and nllb() is not None and src in NLLB_CODES:
                def run_nllb():
                    segs = [dict(x) for x in src_segs]
                    for seg, text in zip(segs, nllb().translate([x["text"] for x in segs], src)):
                        seg["source_text"], seg["text"] = seg["text"], text
                    nllb_res["segs"] = segs
                job = threading.Thread(target=run_nllb, daemon=True)
                job.start()
            if "whisper" in tracks or job is None:
                w_segs, _ = self.decode(audio, src, "translate", (0.0, 0.4))
                out["whisper"] = w_segs
            if job is not None:
                job.join()
                out["nllb"] = nllb_res.get("segs", [])
        return {"tracks": {k: v for k, v in out.items() if k in tracks or k == "nllb" and "nllb" in out},
                "speech_ratio": ratio, "language": src}

    def _transcribe(self, audio, language="ko", task="translate", mt="whisper"):
        """language None = detect. task "translate" (to English) or "transcribe".

        mt="nllb" translates by transcribing in the source language and
        running NLLB-200 on the text (segment timestamps come from Whisper);
        English sources, and languages NLLB lacks, use Whisper alone.
        """
        ratio = speech_ratio(audio)
        use_nllb = task == "translate" and mt == "nllb" and nllb() is not None

        def run(lang, temps):
            if not use_nllb:
                segs, info = self.decode(audio, lang, task, temps)
                return segs, info, "whisper"
            segs, info = self.decode(audio, lang, "transcribe", temps)
            src = lang or info.get("language")
            if src == "en":
                return segs, info, "none"
            if src not in NLLB_CODES:
                segs, info = self.decode(audio, src, "translate", temps)
                return segs, info, "whisper"
            for seg, text in zip(segs, nllb().translate([x["text"] for x in segs], src)):
                seg["source_text"], seg["text"] = seg["text"], text
            return segs, info, "nllb"

        segs, info, used = run(language, (0.0, 0.4))
        if ratio is not None and ratio >= self.RETRY_SPEECH and all(junk(s["text"]) for s in segs):
            log.info("retrying chunk (speech %.0f%%, got %r)", ratio * 100,
                     [s["text"] for s in segs])
            segs, info, used = run(language or info.get("language"), (0.4, 0.8))
        return {"segments": segs, "speech_ratio": ratio, "mt": used, **info}


# Whisper language code -> NLLB-200 code (languages the channels are likely to use).
NLLB_CODES = {
    "ar": "arb_Arab", "bn": "ben_Beng", "cs": "ces_Latn", "de": "deu_Latn", "el": "ell_Grek",
    "en": "eng_Latn", "es": "spa_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn",
    "he": "heb_Hebr", "hi": "hin_Deva", "hu": "hun_Latn", "id": "ind_Latn", "it": "ita_Latn",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "nl": "nld_Latn", "pl": "pol_Latn", "ps": "pbt_Arab",
    "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sv": "swe_Latn", "ta": "tam_Taml",
    "th": "tha_Thai", "tr": "tur_Latn", "uk": "ukr_Cyrl", "ur": "urd_Arab", "vi": "vie_Latn",
    "zh": "zho_Hans",
}


class NLLB:
    """NLLB-200 text translation to English via CTranslate2 (CPU)."""

    def __init__(self, path, threads=4):
        import ctranslate2
        import sentencepiece
        # NLLB_DEVICE=cuda on an NVIDIA ASR server; the relay's guide
        # translation runs on the CPU.
        device = os.environ.get("NLLB_DEVICE", "cpu")
        self.tr = ctranslate2.Translator(path, device=device, compute_type="int8" if device == "cpu" else "float16",
                                         intra_threads=threads)
        self.sp = sentencepiece.SentencePieceProcessor(
            model_file=os.path.join(path, "sentencepiece.bpe.model"))

    def translate(self, texts, src):
        idx = [i for i, t in enumerate(texts) if t.strip()]
        out = list(texts)
        if not idx:
            return out
        batch = [[NLLB_CODES[src]] + self.sp.encode(texts[i].strip(), out_type=str) + ["</s>"]
                 for i in idx]
        res = self.tr.translate_batch(batch, target_prefix=[["eng_Latn"]] * len(batch),
                                      beam_size=2, max_decoding_length=200,
                                      repetition_penalty=1.2, no_repeat_ngram_size=4)
        for i, r in zip(idx, res):
            out[i] = " " + self.sp.decode(r.hypotheses[0][1:])
        return out


BATCHES = collections.Counter()   # (model, batch size) -> batches decoded (asr_server /metrics)


class Batcher:
    """Runs fn(items) -> results over the requests waiting, up to max_size
    with the same key at a time: requests that arrive while a batch runs go
    together into the next one, and a lone request never waits. Decoding
    one chunk at a time, token by token, kept an NVIDIA GPU half idle."""

    def __init__(self, name, fn, max_size, key=lambda item: None):
        self.name, self.fn, self.max, self.key = name.split("-")[0], fn, max(1, max_size), key
        self.cv = threading.Condition()
        self.jobs = []   # [item, Event, result or exception]
        threading.Thread(target=self.loop, name=name, daemon=True).start()

    def run(self, item):
        job = [item, threading.Event(), None]
        with self.cv:
            self.jobs.append(job)
            self.cv.notify()
        job[1].wait()
        if isinstance(job[2], Exception):
            raise job[2]
        return job[2]

    def loop(self):
        while True:
            with self.cv:
                while not self.jobs:
                    self.cv.wait()
                k = self.key(self.jobs[0][0])
                batch = [j for j in self.jobs if self.key(j[0]) == k][:self.max]
                for j in batch:
                    self.jobs.remove(j)
            BATCHES[(self.name, len(batch))] += 1
            try:
                results = self.fn([j[0] for j in batch])
            except Exception as e:  # noqa: BLE001 - every request in the batch gets it
                results = [e] * len(batch)
            for j, r in zip(batch, results):
                j[2] = r
                j[1].set()


def batch_size(model):
    """Most requests decoded together: <MODEL>_BATCH, else ASR_BATCH, else 8."""
    return max(1, int(os.environ.get(f"{model.upper()}_BATCH") or os.environ.get("ASR_BATCH") or 8))


class Seamless:
    """SeamlessM4T v2 speech -> English text (PyTorch; MPS on Apple Silicon).
    Requests are decoded in batches (Batcher, SEAMLESS_BATCH)."""

    def __init__(self, path):
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
        if torch.backends.mps.is_available():
            self.device, dtype = "mps", torch.float16
        elif torch.cuda.is_available():
            self.device, dtype = "cuda", torch.float16
        else:
            self.device, dtype = "cpu", torch.float32
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(path, torch_dtype=dtype).to(self.device).eval()
        self.dtype = dtype
        self.batcher = Batcher("seamless-batch", self.decode, batch_size("seamless"))

    def translate(self, audio):
        return self.batcher.run(audio)

    def decode(self, audios):
        with self.torch.inference_mode():
            inputs = self.proc(audio=audios, sampling_rate=SR, return_tensors="pt", padding=True)
            inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                      for k, v in inputs.items()}
            ids = self.model.generate(**inputs, tgt_lang="eng", max_new_tokens=MAX_TOKENS)
        return [self.proc.decode(x.tolist(), skip_special_tokens=True).strip() for x in ids]


_seamless, _seamless_lock = None, threading.Lock()


def seamless_path():
    return os.environ.get("SEAMLESS_MODEL", os.path.expanduser("~/captions-asr/seamless-m4t-v2-large"))


def seamless():
    """The shared Seamless model, or None if SEAMLESS_MODEL isn't installed."""
    global _seamless
    with _seamless_lock:
        if (_seamless is None and os.environ.get("ASR_BACKEND") != "cpu"
                and os.path.exists(os.path.join(seamless_path(), "config.json"))):
            log.info("loading Seamless from %s", seamless_path())
            _seamless = Seamless(seamless_path())
        return _seamless


_nllb, _nllb_lock = None, threading.Lock()


def nllb():
    """The shared NLLB translator, or None if NLLB_MODEL isn't installed."""
    global _nllb
    path = os.environ.get("NLLB_MODEL", "/models/nllb-200-distilled-1.3B-ct2")
    with _nllb_lock:
        if _nllb is None and os.path.exists(os.path.join(path, "model.bin")):
            log.info("loading NLLB from %s", path)
            _nllb = NLLB(path, int(os.environ.get("NLLB_THREADS", "4")))
        return _nllb


def speech_ratio(audio):
    """Fraction of the chunk Silero VAD calls speech, or None if unavailable."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:
        return None
    ts = get_speech_timestamps(audio, VadOptions(threshold=0.35, min_silence_duration_ms=500))
    return round(sum(t["end"] - t["start"] for t in ts) / max(1, len(audio)), 3)


class CudaWhisper(Whisper):
    """faster-whisper on an NVIDIA GPU (ASR_BACKEND=cuda, for asr_server.py on
    a Linux box; tested on an RTX 3060, see docs/NVIDIA.md), or on the CPU in
    int8 (ASR_BACKEND=cpu, ASR_THREADS). The model is loaded once."""

    _model = None
    _lock = threading.Lock()

    def __init__(self, model="large-v3", beam=1, device="cuda"):
        self.name, self.beam, self.device = model, beam, device

    def _get(self):
        with self._lock:
            if CudaWhisper._model is None:
                from faster_whisper import WhisperModel
                log.info("loading model %s", self.name)
                CudaWhisper._model = WhisperModel(
                    self.name, device=self.device, compute_type="float16" if self.device == "cuda" else "int8",
                    cpu_threads=int(os.environ.get("ASR_THREADS", "0")),
                    # decodes side by side (batches, and the unbatched retries);
                    # more than 2 only made them fight over the GPU (RTX 3060)
                    num_workers=max(1, int(os.environ.get("WHISPER_WORKERS", "2"))),
                    download_root=os.environ.get("MODEL_DIR", "/models"))
                log.info("model ready")
            return CudaWhisper._model

    _batcher = None

    def decode(self, audio, language, task, temperatures):
        # Known language, first (greedy) try: batched with other requests of
        # the same language and task. A looping decode falls back to the
        # warmer temperatures one at a time, as does language detection.
        if language and temperatures and temperatures[0] == 0 and batch_size("whisper") > 1:
            with self._lock:
                if CudaWhisper._batcher is None:
                    CudaWhisper._batcher = Batcher("whisper-batch", self.decode_batch, batch_size("whisper"),
                                                   key=lambda item: item[1:])
            segs = CudaWhisper._batcher.run((audio, language, task))
            if segs is not None:
                return segs, {"language": language, "language_probability": None}
            temperatures = temperatures[1:] or (0.4,)
        segs, info = self._get().transcribe(
            audio, language=language, task=task, beam_size=self.beam,
            temperature=list(temperatures), compression_ratio_threshold=2.4,
            vad_filter=False, condition_on_previous_text=False,
            no_speech_threshold=None, log_prob_threshold=None, max_new_tokens=MAX_TOKENS)
        segs = [{"start": s.start, "end": s.end, "text": s.text,
                 "no_speech_prob": s.no_speech_prob, "avg_logprob": s.avg_logprob,
                 "compression_ratio": s.compression_ratio} for s in segs]
        return segs, {"language": info.language,
                      "language_probability": round(info.language_probability, 3)}


    def decode_batch(self, items):
        """[(audio, language, task)] of one language and task -> segments for
        each, or None where the text loops (compression ratio > 2.4)."""
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_compression_ratio, get_suppressed_tokens
        m = self._get()
        _, language, task = items[0]
        tok = Tokenizer(m.hf_tokenizer, m.model.is_multilingual, task=task, language=language)
        enc = m.encode(np.stack([pad_or_trim(m.feature_extractor(a)) for a, _, _ in items]))
        prompt = m.get_prompt(tok, [], without_timestamps=False)
        results = m.model.generate(
            enc, [prompt] * len(items), beam_size=self.beam, max_length=len(prompt) + MAX_TOKENS,
            return_scores=True, return_no_speech_prob=True, suppress_blank=True,
            suppress_tokens=get_suppressed_tokens(tok, [-1]),
            max_initial_timestamp_index=int(round(1.0 / m.time_precision)))
        out = []
        for (audio, _, _), r in zip(items, results):
            tokens = r.sequences_ids[0]
            text = tok.decode([t for t in tokens if t < tok.eot]).strip()
            ratio = get_compression_ratio(text)
            if ratio > 2.4:
                out.append(None)
                continue
            avg = r.scores[0] * len(tokens) / (len(tokens) + 1)
            dur, segs, start, buf = len(audio) / SR, [], 0.0, []
            for t in tokens + [tok.timestamp_begin + round(dur / m.time_precision)]:
                if t >= tok.timestamp_begin:   # a timestamp: closes the text before it
                    ts = min(dur, (t - tok.timestamp_begin) * m.time_precision)
                    if buf:
                        segs.append({"start": start, "end": max(ts, start), "text": tok.decode(buf),
                                     "no_speech_prob": r.no_speech_prob, "avg_logprob": avg,
                                     "compression_ratio": ratio})
                        buf = []
                    start = ts
                elif t < tok.eot:
                    buf.append(t)
            out.append(segs)
        return out


class MlxWhisper(Whisper):
    """mlx-whisper on Apple Silicon (Metal). Used by the Mac ASR service."""

    def __init__(self, model="mlx-community/whisper-large-v3-mlx"):
        self.model = model
        self.lock = threading.Lock()  # one decode at a time on the GPU

    def decode(self, audio, language, task, temperatures):
        import mlx_whisper
        with self.lock:
            r = mlx_whisper.transcribe(
                audio, path_or_hf_repo=self.model, task=task, language=language,
                temperature=tuple(temperatures), compression_ratio_threshold=2.4,
                condition_on_previous_text=False, sample_len=MAX_TOKENS,
                no_speech_threshold=None, logprob_threshold=None, verbose=None)
        segs = [{"start": s["start"], "end": s["end"], "text": s["text"],
                 "no_speech_prob": s.get("no_speech_prob"), "avg_logprob": s.get("avg_logprob"),
                 "compression_ratio": s.get("compression_ratio")} for s in r["segments"]]
        return segs, {"language": r.get("language"), "language_probability": None}


def wav_bytes(audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return buf.getvalue()


class Remote:
    """HTTP backend; see README "ASR HTTP contract"."""

    def __init__(self, url, timeout=15):
        self.url, self.timeout = url.rstrip("/"), timeout
        self._api, self._api_at = None, 0.0
        self.mts = ["whisper"]   # translation methods the server offers (from /health)

    def api(self):
        """The server's contract version from /health (1 if it doesn't say)."""
        if time.monotonic() - self._api_at > 60:
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=3) as r:
                    h = json.load(r)
                    self._api = int(h.get("api", 1))
                    self.mts = list(h.get("mt", ["whisper"]))
            except Exception:  # noqa: BLE001 - unknown; the request itself will tell
                self._api = self._api or 1
            self._api_at = time.monotonic()
        return self._api

    def transcribe(self, audio, language="ko", task="translate", mt="whisper", timeout=None, due=None):
        if self.api() < 2 and (language != "ko" or task != "translate"):
            # Version 1 servers only did Korean -> English.
            raise Unsupported("remote ASR is API v1 (Korean translate only)")
        q = f"language={language or 'auto'}&task={task}&mt={mt}"
        req = urllib.request.Request(
            f"{self.url}/v1/translate?{q}", data=wav_bytes(audio), headers=_headers(due), method="POST")
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            r = json.load(r)
            return {"segments": r["segments"], "speech_ratio": r.get("speech_ratio"),
                    "language": r.get("language"), "mt": r.get("mt", "whisper"),
                    "language_probability": r.get("language_probability")}


def _headers(due):
    """Request headers; due (time.monotonic()) is when the caption is needed,
    for the relay's own ASR queue (relay.py, per-channel worker processes)."""
    return {"Content-Type": "audio/wav", **({"X-Due": f"{due:.3f}"} if due else {})}


def remote_multi(self, audio, language, tracks, timeout=None, due=None):
    """POST /v1/multi (API v4): every track for one chunk in one request."""
    q = f"language={language or 'auto'}&tracks={','.join(tracks)}"
    req = urllib.request.Request(f"{self.url}/v1/multi?{q}", data=wav_bytes(audio),
                                 headers=_headers(due), method="POST")
    with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
        return json.load(r)


Remote.multi = remote_multi


class Unsupported(Exception):
    """The backend can't do this request; another backend should."""


class Disabled:
    """No ASR: remux only (for measuring the relay on its own)."""

    def transcribe(self, audio, language="ko", task="translate", mt="whisper"):
        return {"segments": [], "speech_ratio": None}


def from_env():
    """The relay: remote (an ASR server over HTTP) or none (no subtitles).
    The ASR server (asr_server.py): mlx (Apple Silicon) or cuda (NVIDIA)."""
    kind = os.environ.get("ASR_BACKEND") or ("remote" if os.environ.get("ASR_URL") else "none")
    if kind == "mlx":
        return MlxWhisper(os.environ.get("MODEL", "mlx-community/whisper-large-v3-mlx"))
    if kind == "cuda":
        return CudaWhisper(os.environ.get("MODEL", "large-v3"), int(os.environ.get("BEAM", "1")))
    if kind == "cpu":
        return CudaWhisper(os.environ.get("MODEL", "large-v3-turbo"), int(os.environ.get("BEAM", "1")), "cpu")
    if kind == "remote":
        return Remote(os.environ["ASR_URL"], float(os.environ.get("ASR_TIMEOUT", "15")))
    if kind == "none":
        return Disabled()
    raise ValueError(f"unknown ASR_BACKEND {kind}")


def timed(backend, audio, language="ko", task="translate", mt="whisper"):
    t0 = time.monotonic()
    r = backend.transcribe(audio, language, task, mt)
    return r, time.monotonic() - t0
