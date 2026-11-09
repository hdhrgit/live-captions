"""captions-dvr relay: HLS in, MPEG-TS with CEA-608 English captions out.

Each channel runs on demand: the first client starts a session, and the
session stops GRACE seconds after the last client leaves. A session has
three threads:
  input  - demuxes the HLS source, normalizes timestamps, queues packets for
           output and decodes audio for ASR
  asr    - cuts audio at quiet points, sends it to the ASR backend and turns
           guarded segments into CEA-608 cues
  output - paces packets out about DELAY seconds behind the source, inserts
           caption SEI into each video access unit (no re-encode) and muxes
           MPEG-TS
The HTTP server serves /<id>.ts per channel, /playlist.m3u and /status.
"""
import collections
import contextlib
import heapq
import io
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import re
import time
import urllib.parse
import urllib.request
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import av
import numpy as np

import adhoc
import asr
import burn
import cc608
import cc608dec
import config
import epg
import health
import hls
import slate

DELAY = float(os.environ.get("DELAY", "20"))
MIN_LEAD = float(os.environ.get("MIN_LEAD", "15"))
# Output starts this long after the first packet: time for the first chunk's
# captions (the slate card covers the wait), and extra ASR lead from the start.
START_DELAY = float(os.environ.get("START_DELAY", "8"))
# On tune, release this much of the output at once (out of START_DELAY), so
# the player has a few seconds buffered when the real picture follows the
# slate card; otherwise the first pieces arrive just in time and it stalls
# right after the first frame. The input starts far enough back for the
# delay plus this when the source's playlist window allows.
BURST = float(os.environ.get("BURST", "0"))
# An ffmpeg binary that can decode audio this PyAV can't (AC-4, ATSC 3.0):
# sources with such audio are read through it (Session.open_source). Not
# bundled; set it here or in the web page ("Audio ffmpeg"), which wins.
AC4_FFMPEG = os.environ.get("AC4_FFMPEG", "")
COLD_WAIT = float(os.environ.get("COLD_WAIT", "8"))   # max hold of a playlist request with no segments yet
GRACE = float(os.environ.get("GRACE", "60"))
PORT = int(os.environ.get("PORT", "8095"))
PUBLIC_URL = os.environ.get("PUBLIC_URL", f"http://localhost:{PORT}")
SEED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "channels.json")
CHANNELS_FILE = os.environ.get("CHANNELS_FILE", SEED_FILE)
# ASR chunks: cut at the first speech pause (PAUSE_MS) after CHUNK_MIN
# seconds, or at the quietest point once CHUNK_MAX is reached.
CHUNK_MIN = float(os.environ.get("CHUNK_MIN", "4"))
CHUNK_MAX = float(os.environ.get("CHUNK_MAX", "15"))
PAUSE_MS = int(os.environ.get("PAUSE_MS", "300"))
# A sentence left unfinished at the end of a chunk waits for the next chunk
# only while its cue would still be at least this far ahead of the output.
HOLD_MIN_LEAD = float(os.environ.get("HOLD_MIN_LEAD", "15"))
STALL = float(os.environ.get("STALL", "60"))
# 708 centring mode when a channel doesn't set one. Channels DVR showed the
# WebVTT subtitles with 608 + 708 embedded as well, so keep both by default.
CC_CENTER = os.environ.get("CC_CENTER", "nbtsp")
VTT_CHARS = int(os.environ.get("VTT_CHARS", "42"))   # WebVTT line length (0 = no breaks)
# Embed CEA-608/708 in the video as well as the WebVTT track. Off by default:
# Channels shows the WebVTT subtitles without it, and EIA added a second
# (left-aligned on tvOS) caption track.
EIA_DEFAULT = os.environ.get("EIA", "0") == "1"
ARCHIVE_DIR = os.environ.get("ARCHIVE_DIR", "")
# Languages detected on "auto" channels, per source URL, so a new session
# starts with the right one: {url: {"lang", "votes", "at"}}.
LANG_FILE = os.environ.get("LANG_FILE", os.path.join(os.path.dirname(os.path.abspath(CHANNELS_FILE)),
                                                     "languages.json"))
LID_EVERY = float(os.environ.get("LID_EVERY", "1800"))   # re-check the language this often (s)
LID_AT_START = 3                                          # and on this many speech chunks per session


class Languages:
    """The detected-language store (LANG_FILE)."""

    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        try:
            with open(path) as f:
                self.data = json.load(f)
        except (OSError, ValueError):
            self.data = {}

    def get(self, url):
        with self.lock:
            return dict(self.data.get(url) or {})

    def put(self, url, lang, votes):
        with self.lock:
            self.data[url] = {"lang": lang, "votes": list(votes), "at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
            self.save()

    def forget(self, url):
        with self.lock:
            self.data.pop(url, None)
            self.save()

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=1)
            os.replace(tmp, self.path)
        except OSError as e:
            log.warning("can't save %s: %s", self.path, e)


languages = Languages(LANG_FILE)
ARCHIVE_MINUTES = float(os.environ.get("ARCHIVE_MINUTES", "60"))
MUSIC_CAPTION = os.environ.get("MUSIC_CAPTION", "[\u266a music]")

TICKS = 90000
SR = asr.SR
TB = Fraction(1, TICKS)
GPU_LOCK = threading.Lock()   # one request to the remote ASR (Mac GPU) at a time
VIDEO_PID = 0x100  # libavformat mpegts default for the first stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s")
log = logging.getLogger("relay")
backend = asr.from_env()


class Timeline:
    """Maps source timestamps onto one continuous output timeline."""

    def __init__(self):
        self.offset = None
        self.last = None

    def map(self, ts):
        if self.offset is None:
            # From 60 s: below that, a cold start's slate segments (slate.py).
            self.offset = TICKS * 60 - ts
        out = ts + self.offset
        if self.last is not None and abs(out - self.last) > 5 * TICKS:
            log.warning("timestamp jump of %.1fs, re-basing", (out - self.last) / TICKS)
            self.offset = self.last + 3600 - ts
            out = ts + self.offset
        self.last = max(self.last or out, out)
        return out


class Broadcaster(io.RawIOBase):
    """File-like sink for the muxer that fans bytes out to HTTP clients."""

    def __init__(self):
        self.clients = set()
        self.lock = threading.Lock()
        self.total = 0

    def writable(self):
        return True

    def write(self, b):
        data = bytes(b)
        with self.lock:
            for cq in list(self.clients):
                if cq.waiting:
                    start = keyframe_offset(data, (-self.total) % 188)
                    if start is None:
                        continue
                    chunk, cq.waiting = data[start:], False
                else:
                    chunk = data
                try:
                    cq.put_nowait(chunk)
                except queue.Full:
                    self.clients.discard(cq)
                    cq.put(None)
            self.total += len(data)
        return len(data)

    def reset(self):
        """New session: the next bytes start a new mux from scratch."""
        with self.lock:
            self.total = 0
            for cq in self.clients:
                cq.waiting = True


def keyframe_offset(data, pos):
    """Offset of the first TS packet starting a video keyframe PES, or None."""
    while pos + 188 <= len(data):
        if (data[pos] == 0x47 and (data[pos + 1] & 0x40)            # PUSI
                and ((data[pos + 1] & 0x1F) << 8 | data[pos + 2]) == VIDEO_PID
                and (data[pos + 3] & 0x20) and data[pos + 4]          # adaptation
                and (data[pos + 5] & 0x40)):                          # RAI
            return pos
        pos += 188
    return None


class SessionSink(io.RawIOBase):
    """Muxer output for one session; goes quiet once the session stops."""

    def __init__(self, session):
        self.session = session

    def writable(self):
        return True

    def write(self, b):
        if self.session.stop.is_set():
            return len(b)
        return self.session.ch.broadcaster.write(b)


class SkipChunk(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Session:
    """One running pull/caption/mux pipeline for a channel."""

    def __init__(self, channel):
        self.ch = channel
        self.stop = threading.Event()
        self.encoder = cc608.Encoder()
        self.lock = threading.Lock()
        self.guard = asr.Guard()
        self.outq = queue.Queue(maxsize=20000)   # (arrival, kind, packet)
        self.audioq = queue.Queue(maxsize=2000)  # (start_ticks|None, samples) or None
        self.last_input = time.monotonic()
        self.captionable = True
        self.cc_checked = self.cc_seen = 0
        self.source_cc = False
        self.variant_failed = False
        self.url_idx = 0
        self.anchor = None  # output clock (wall, ticks); read by asr_call
        self.compare_guards = {}  # per extra subtitle track
        self.vtt_lock = threading.Lock()  # Seamless cues arrive from another thread
        # Language "auto": the language detected on this source before, else
        # the country's, until the ASR says otherwise.
        self.votes = collections.deque(maxlen=8)
        self.language = None
        if channel.language == "auto":
            known = languages.get(channel.url)
            self.votes.extend(known.get("votes", []))
            self.language = known.get("lang") or epg.guide_language(
                channel.cfg, config.channel_countries(channel.cfg, cfg.sources))
        self.lid_checks, self.lid_last = 0, 0.0
        self.burst = BURST   # seconds of backlog released at once on start
        self.start_delay = START_DELAY   # input_loop: DELAY for sources without a backlog
        # New session, new timeline: cues of an earlier session would hold
        # the new ones back (cue starts must keep increasing).
        for cues in channel.vtt_tracks.values():
            cues.clear()
        self.chunk_no = 0
        self.pending = None   # (a, b, text, chunk_no): unfinished sentence held for the next chunk
        self.stats = {"input": "starting", "asr_rtf": None, "asr_backlog": 0,
                      "lead_s": None, "captions": 0, "dropped": {}}
        self.threads = [threading.Thread(target=fn, name=f"{channel.id}-{fn.__name__}", daemon=True)
                        for fn in (self.input_loop, self.asr_loop, self.output_loop)]

    def start(self):
        for t in self.threads:
            t.start()

    # ------------------------------------------------------------ input ---

    def input_loop(self):
        timeline = Timeline()
        while not self.stop.is_set():
            try:
                self.stats["input"] = "connecting"
                headers = self.ch.cfg.get("headers", {})
                options = {"user_agent": headers.get("user_agent") or "Mozilla/5.0",
                           "reconnect": "1", "reconnect_streamed": "1", "reconnect_delay_max": "10",
                           # Start this far back so the backlog covers the delay.
                           "live_start_index": str(self.ch.live_start_index)}
                if headers.get("referer"):
                    options["headers"] = f"Referer: {headers['referer']}\r\n"
                # Feeds of this channel, best first; rotate through them on errors.
                urls = self.ch.urls()
                base = urls[self.url_idx % len(urls)]
                self.stats["source_url"] = base
                # If the picked variant failed last time, let libavformat open the master.
                url = base if self.variant_failed else pick_variant(base, headers)
                self.stats["variant"] = url if url != base else None
                if self.anchor is None and ".m3u" not in url.split("?")[0]:
                    # A plain TS stream (e.g. an HDHomeRun tuner) has no
                    # backlog: build the whole delay up front, under the card.
                    self.start_delay = DELAY
                if "live_start_index" not in self.ch.cfg and self.anchor is None:
                    # Start as far back as the delay plus the burst needs.
                    durs = segment_durations(url, headers)
                    if durs:
                        seg = max(durs)
                        k, back = 0, 0.0
                        while k < len(durs) and back < DELAY + BURST + seg:
                            k += 1
                            back += durs[-k]
                        options["live_start_index"] = str(-k)
                        self.stats["backlog_s"] = round(back, 1)
                with self.open_source(url, options) as c:
                    v = c.streams.video[0]
                    a = c.streams.audio[0]
                    self.stats["video_codec"] = v.codec_context.name
                    # For a slate in the source's format next time (slate.py).
                    self.ch.source_format = {
                        "width": v.codec_context.width or 1920, "height": v.codec_context.height or 1080,
                        "fps": int(round(float(v.average_rate or 25))),
                        "sample_rate": a.codec_context.sample_rate or 48000,
                        "channels": a.codec_context.channels or 2,
                        "profile": v.codec_context.profile or "High",
                        "vcodec": v.codec_context.name if v.codec_context.name in ("mpeg2video", "hevc") else "h264",
                        "acodec": "ac3" if a.codec_context.name in ("ac3", "eac3") else "aac"}
                    slate.remember(self.ch.id, self.ch.source_format)
                    self.captionable = v.codec_context.name == "h264"
                    if not self.captionable:
                        log.warning("%s: %s video, relaying without captions",
                                    self.ch.id, v.codec_context.name)
                    resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
                    self.stats["input"] = "streaming"
                    log.info("%s: input open %s", self.ch.id, self.ch.url)
                    for pkt in c.demux(v, a):
                        if self.stop.is_set():
                            return
                        if pkt.dts is None or pkt.size == 0:
                            continue
                        if pkt.time_base != TB:
                            # Normalise everything to 90 kHz ticks.
                            tb = pkt.time_base
                            pkt.dts = int(pkt.dts * tb * TICKS)
                            pkt.pts = int(pkt.pts * tb * TICKS) if pkt.pts is not None else None
                            pkt.duration = int(pkt.duration * tb * TICKS) if pkt.duration else 0
                            pkt.time_base = TB
                        shift = timeline.map(pkt.dts) - pkt.dts
                        pkt.dts += shift
                        pkt.pts = pkt.pts + shift if pkt.pts is not None else pkt.dts
                        kind = "v" if pkt.stream is v else "a"
                        if kind == "a":
                            # Decode before queueing: muxing consumes the packet.
                            for frame in pkt.decode():
                                # The packet's timestamps are already in shifted 90 kHz ticks.
                                start = frame.pts
                                for rf in resampler.resample(frame):
                                    try:
                                        self.audioq.put_nowait((start, rf.to_ndarray()[0].copy()))
                                    except queue.Full:
                                        pass
                                    start = None  # only the first resampled frame is anchored
                        self.outq.put((time.monotonic(), kind, pkt))
                        self.last_input = time.monotonic()
                    raise RuntimeError("source ended")
            except Exception as e:  # noqa: BLE001 - keep the channel alive
                self.stats["input"] = f"error: {e}"
                if self.stats.get("variant") and not self.variant_failed:
                    self.variant_failed = True       # retry the same feed via its master
                else:
                    self.variant_failed = False
                    self.url_idx += 1                # then fail over to the next feed
                log.warning("%s: input error: %s; retrying in 5s", self.ch.id, e)
                try:
                    self.audioq.put_nowait(None)
                except queue.Full:
                    pass
                self.stop.wait(5)

    # -------------------------------------------------------------- asr ---

    @contextlib.contextmanager
    def open_source(self, url, options):
        """The source as a PyAV container. Audio that this PyAV can't decode
        (AC-4, e.g. ATSC 3.0) would crash the muxer: such a source is read
        through AC4_FFMPEG instead (Channels DVR's ffmpeg has the decoder),
        video copied and audio converted to AAC for both the DVR and ASR."""
        proc = None
        if not getattr(self.ch, "needs_audio_convert", False):
            c = av.open(url, timeout=(20, 20), options=options)
            if all(s.codec_context is not None for s in c.streams.audio[:1]):
                try:
                    yield c
                finally:
                    c.close()
                return
            c.close()
            self.ch.needs_audio_convert = True
        ffmpeg = cfg.settings.get("audio_ffmpeg") or AC4_FFMPEG
        if not ffmpeg:
            raise RuntimeError("source audio can't be decoded (AC-4?): set an audio ffmpeg (AC4_FFMPEG)")
        log.info("%s: converting the source's audio with %s", self.ch.id, ffmpeg)
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-user_agent", options.get("user_agent", "Mozilla/5.0"),
               "-i", url, "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy", "-c:a", "aac", "-ac", "2", "-b:a", "192k",
               "-f", "mpegts", "pipe:1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
        try:
            c = av.open(proc.stdout, format="mpegts", timeout=(20, 20))
            try:
                yield c
            finally:
                c.close()
        finally:
            proc.kill()
            proc.wait()

    def asr_loop(self):
        buf, buf_start, buf_len = [], None, 0
        while not self.stop.is_set():
            try:
                item = self.audioq.get(timeout=1)
            except queue.Empty:
                continue
            self.stats["asr_backlog"] = self.audioq.qsize()
            if item is None:
                buf, buf_start, buf_len = [], None, 0
                continue
            start, samples = item
            if start is not None and buf_start is not None:
                expected = buf_start + buf_len * TICKS // SR
                if abs(start - expected) > TICKS // 5:  # gap: restart buffer
                    buf, buf_start, buf_len = [], None, 0
            if buf_start is None:
                if start is None:
                    continue
                buf_start = start
            buf.append(samples)
            buf_len += len(samples)
            if buf_len < CHUNK_MIN * SR:
                continue
            audio = np.concatenate(buf)
            cut = find_cut(audio)
            if cut is None:
                buf = [audio]
                continue
            chunk_start = buf_start
            rest = audio[cut:]
            buf, buf_start, buf_len = [rest], buf_start + cut * TICKS // SR, len(rest)
            if self.audioq.qsize() > 1000:  # way behind: skip rather than drift further
                log.warning("%s: asr backlog, dropping %.1fs chunk", self.ch.id, cut / SR)
                continue
            self.caption(audio[:cut], chunk_start)

    def detect(self, lang, prob, ratio):
        """Language "auto": majority of the last 8 confident speech chunks
        (kept across sessions in LANG_FILE).

        Starts from the language detected on this source before, else the
        channel's country language, takes the first language two chunks agree
        on if there is none, and switches only when 6 of the last 8 agree on
        another one.
        """
        if not lang or (ratio or 0) < 0.5 or (prob is not None and prob < 0.6):
            return
        self.votes.append(lang)
        top, n = collections.Counter(self.votes).most_common(1)[0]
        # Undecided after ~8 chunks (about a minute, e.g. a start in music):
        # one confident speech chunk is enough.
        if self.language is None and (n >= 2 or n >= 1 and self.chunk_no >= 8) or top != self.language and n >= 6:
            log.info("%s: language %s -> %s (%d of last %d chunks)", self.ch.id, self.language,
                     top, n, len(self.votes))
            self.language = top
        if self.language:
            languages.put(self.ch.url, self.language, self.votes)

    def check_language(self, audio, start, ratio):
        """Whisper language ID in the background (Seamless reports none): on
        the first speech chunks of a session, then every LID_EVERY seconds."""
        try:
            r, _ = self.asr_call(audio, start, None, "transcribe", "whisper")
        except Exception:  # noqa: BLE001 - try again on a later chunk
            return
        self.detect(r.get("language"), r.get("language_probability"),
                    r.get("speech_ratio") if r.get("speech_ratio") is not None else ratio)

    def drop(self, reason):
        self.stats["dropped"][reason] = self.stats["dropped"].get(reason, 0) + 1

    def asr_call(self, audio, start, lang, task, mt):
        """Run ASR; for a remote backend, retry while the caption can still be on time.

        The chunk's first frame goes out at its release time (the output
        clock). Until shortly before that, a busy (503) or unreachable remote
        is retried with backoff; after it the chunk is skipped uncaptioned.
        """
        if not isinstance(backend, asr.Remote):
            return asr.timed(backend, audio, lang, task, mt)
        anchor = self.anchor
        due = (anchor[0] + (start - anchor[1]) / TICKS) if anchor else time.monotonic() + DELAY / 2
        t0, pause, reason = time.monotonic(), 0.5, "asr_down"
        while True:
            remaining = due - time.monotonic() - 1.0
            if remaining < 0.5:
                log.info("%s: skipping chunk, remote ASR %s", self.ch.id,
                         "busy" if reason == "asr_busy" else "unreachable")
                raise SkipChunk(reason)
            try:
                # One Mac request at a time across channels: its GPU decodes
                # serially anyway, and concurrent GPU jobs ran away before.
                if not GPU_LOCK.acquire(timeout=max(0.1, remaining - 1)):
                    raise TimeoutError("waiting for the GPU lane")
                try:
                    r = self.call_backend(audio, lang, task, mt, min(remaining, backend.timeout))
                finally:
                    GPU_LOCK.release()
                r["backend"] = "remote"
                return r, time.monotonic() - t0
            except asr.Unsupported as e:
                raise SkipChunk("asr_unsupported") from e
            except Exception as e:  # noqa: BLE001 - busy, timed out or unreachable: retry
                reason = "asr_busy" if getattr(e, "code", None) == 503 else "asr_down"
                self.stats["asr_last_error"] = f"{type(e).__name__}: {e}"
                time.sleep(min(pause, max(0.0, due - time.monotonic() - 1.5)))
                pause = min(pause * 2, 2.0)

    @staticmethod
    def call_backend(audio, lang, task, mt, timeout):
        if isinstance(mt, list):   # API v4: every track in one request
            m = backend.multi(audio, lang, mt, timeout=timeout)
            return {"tracks": m["tracks"], "speech_ratio": m.get("speech_ratio"),
                    "language": m.get("language"), "errors": m.get("errors", {})}
        return backend.transcribe(audio, lang, task, mt, timeout=timeout)

    def watch_source_cc(self, data):
        """Notice CEA-608/708 captions already in the source (first 500 frames)."""
        if self.cc_checked >= 500:
            return
        self.cc_checked += 1
        if b"GA94" in data:
            self.cc_seen += 1
        if self.cc_checked in (100, 500):
            self.source_cc = self.cc_seen > self.cc_checked // 5
            self.stats["source_captions"] = self.source_cc
            self.stats["captions_mode"] = ("source (passed through)" if self.use_source_cc
                                           else "generated")
            if self.cc_checked == 100 and self.source_cc:
                log.info("%s: source has its own captions; %s", self.ch.id,
                         "passing them through" if self.use_source_cc else "replacing them")

    # Read live, so switching a channel to "generate" in the web page takes
    # effect in the running session.
    # "source": keep the source's own captions (CC1, also decoded into the
    # WebVTT track) and skip ASR. Default "generate": ours, the source's dropped.
    use_source_cc = property(lambda self: self.source_cc and self.ch.cfg.get("captions") == "source")

    def caption(self, audio, start):
        if self.use_source_cc:
            return self.drop("source_captions")
        rec = {"pts": round(start / TICKS, 2), "wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "duration": round(len(audio) / SR, 2)}
        self.chunk_no += 1
        try:
            self._caption(audio, start, rec)
        finally:
            if self.pending and self.pending[3] < self.chunk_no:
                # Held sentence start, but this chunk had nothing to add.
                a, b, text, _ = self.pending
                self.pending = None
                self.emit(a, b, text)
            if ARCHIVE_DIR:
                self.archive(audio, rec)

    def _caption(self, audio, start, rec):
        reason = rec["gate"] = asr.gate(audio)
        if reason:
            return self.drop(reason)
        lang = self.ch.language if self.ch.language != "auto" else self.language
        if lang and lang not in asr.WHISPER_LANGS:
            self.stats["language"] = f"{lang} (not supported by Whisper)"
            return self.drop("lang_unsupported")
        task = "transcribe" if lang == "en" else "translate"
        # "auto" channels let the ASR detect every chunk, so a wrong guess
        # (e.g. an English clip on a Hebrew channel) can't stick.
        asr_lang = None if self.ch.language == "auto" else lang
        rec["language"], rec["task"] = lang, task
        mt = hls.main_mt(self.ch.cfg)
        if mt == "seamless" and self.ch.language == "auto" and lang is None:
            # Language unknown (no country, e.g. stateless mode): Whisper
            # first, which detects it; Seamless once two chunks agree.
            mt = "whisper"
        if task == "transcribe" and mt == "seamless":
            mt = "whisper"   # English source: Whisper transcribes; Seamless eng->eng adds nothing
        compare = self.ch.cfg.get("compare", False)
        try:
            if mt == "seamless":
                # One Seamless request; comparison tracks (if enabled) follow
                # in their own thread, so the main track never waits for them.
                r, el = self.asr_call(audio, start, asr_lang, task, "seamless")
            elif (task == "translate" and isinstance(backend, asr.Remote) and backend.api() >= 4
                    and compare):
                # One request for all tracks; the Mac overlaps the work.
                chosen = hls.compare_mts(self.ch.cfg)
                wanted = [mt] + [t for t in ("nllb", "whisper") if t != mt and t in chosen] + ["original"]
                r, el = self.asr_call(audio, start, asr_lang, task, wanted)
                if "seamless" in backend.mts and "seamless" in chosen:
                    # After the main tracks, in its own thread: the GPU stays
                    # serial (GPU_LOCK) and nothing waits for Seamless.
                    threading.Thread(target=self.seamless_track, args=(audio, start, asr_lang),
                                     name=f"{self.ch.id}-seamless", daemon=True).start()
                r["segments"] = r["tracks"].get(mt, [])
                r["mt"] = mt
            else:
                r, el = self.asr_call(audio, start, asr_lang, task, mt)
        except SkipChunk as e:
            rec["error"] = str(e)
            return self.drop(e.reason)
        except Exception as e:  # noqa: BLE001 - a failed chunk is just uncaptioned
            log.warning("%s: asr failed: %s", self.ch.id, e)
            rec["error"] = str(e)
            return self.drop("asr_error")
        self.stats["asr_rtf"] = round(el / (len(audio) / SR), 3)
        self.stats["asr_last_s"] = round(el, 2)
        self.stats["asr_used"] = rec["backend"] = r.get("backend", type(backend).__name__)
        ratio = rec["speech_ratio"] = r.get("speech_ratio")
        if self.ch.language == "auto":
            if r.get("mt") == "seamless":
                if (ratio or 0) >= 0.5 and isinstance(backend, asr.Remote) and (
                        self.lid_checks < LID_AT_START or time.monotonic() - self.lid_last > LID_EVERY):
                    self.lid_checks += 1
                    self.lid_last = time.monotonic()
                    threading.Thread(target=self.check_language, args=(audio, start, ratio),
                                     name=f"{self.ch.id}-lid", daemon=True).start()
            else:
                self.detect(r.get("language"), r.get("language_probability"), ratio)
        self.stats["language"] = lang or "detecting"
        self.stats["mt"] = r.get("mt")
        self.stats["task"] = task
        rec["segments"] = r["segments"]
        kept = []
        for s in r["segments"]:
            reason = s["drop"] = ((asr.seamless_check(s, ratio) if r.get("mt") == "seamless" else None)
                                  or self.guard.keep(s, ratio))
            a, b = start + int(s["start"] * TICKS), start + int(s["end"] * TICKS)
            if reason:
                log.info("%s: drop (%s, speech %s) [%.1f] %s", self.ch.id, reason, ratio,
                         a / TICKS, s["text"].strip())
                self.drop(reason)
                continue
            # Whisper often splits English into 2-3 word segments; join
            # neighbours into cues of up to two caption rows.
            text = s["text"].strip()
            if (kept and a - kept[-1][1] < TICKS * 0.6 and b - kept[-1][0] < TICKS * 7
                    and len(kept[-1][2]) + 1 + len(text) <= 2 * cc608.COLS - 4):
                kept[-1] = (kept[-1][0], b, kept[-1][2] + " " + text)
            else:
                kept.append((a, b, text))
        if r.get("mt") == "seamless" and kept:
            # One segment for the whole chunk: put the text on the speech.
            regions = [(start + int(s * TICKS), start + int(e * TICKS)) for s, e in asr.speech_regions(audio)]
            kept = [(a, b, t) for a, b, text in kept for a, b, t in self.place(a, b, text, regions)]
        for a, b, text in self.sentence_flow(kept):
            self.emit(a, b, text)
        if compare and r.get("mt") == "seamless":
            if isinstance(backend, asr.Remote) and backend.api() >= 4:
                threading.Thread(target=self.extra_tracks, args=(audio, start, asr_lang, lang, task, ratio),
                                 name=f"{self.ch.id}-compare", daemon=True).start()
        elif compare and task == "transcribe":
            # English source: the main track is Whisper's transcript; compare
            # it with Seamless (English speech to English text).
            if "seamless" in getattr(backend, "mts", ()) and "seamless" in hls.compare_mts(self.ch.cfg):
                threading.Thread(target=self.seamless_track, args=(audio, start, asr_lang),
                                 name=f"{self.ch.id}-seamless", daemon=True).start()
        elif compare:
            t0 = time.monotonic()
            self.compare(audio, start, lang, task, r.get("mt") or mt, r)
            self.stats["compare_total_s"] = round(el + time.monotonic() - t0, 2)
        detecting = self.ch.language == "auto" and self.language is None
        if not kept and MUSIC_CAPTION and (ratio is None or ratio < 0.5) and asr.loud(audio):
            # Not bars or silence (gated above) and no usable words: music.
            # (Music doesn't count toward language detection: detect() needs speech.)
            self.emit(start, start + len(audio) * TICKS // SR,
                      MUSIC_CAPTION + (" - detecting language..." if detecting else ""), log_it=False)
            self.drop("music_caption")
            rec["music_caption"] = True
        elif not kept and detecting:
            self.emit(start, start + len(audio) * TICKS // SR, "detecting language...", log_it=False)

    def archive(self, audio, rec):
        """Keep the chunk's audio and ASR record for ARCHIVE_MINUTES (debugging)."""
        try:
            d = os.path.join(ARCHIVE_DIR, self.ch.id)
            os.makedirs(d, exist_ok=True)
            base = os.path.join(d, rec["wall"].replace(":", "") + f"_{rec['pts']:09.2f}")
            with open(base + ".wav", "wb") as f:
                f.write(asr.wav_bytes(audio))
            with open(base + ".json", "w") as f:
                json.dump(rec, f, ensure_ascii=False, indent=1)
            cutoff = time.time() - ARCHIVE_MINUTES * 60
            for name in os.listdir(d):
                path = os.path.join(d, name)
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
        except OSError as e:
            log.warning("%s: archive failed: %s", self.ch.id, e)

    def add_vtt(self, track, a, b, text, ascii_only=True):
        """Lay out text for a WebVTT track and append it, keeping cue starts
        strictly increasing (the DVR's remuxer rejects backwards timestamps)
        and cues non-overlapping (Fire TV stacks overlapping cues)."""
        cues = self.ch.vtt_tracks.setdefault(track, collections.deque(maxlen=300))
        # The WebVTT tracks have no 32-column limit: their own line length.
        vtt_chars = int(self.ch.cfg.get("vtt_chars", VTT_CHARS))
        vtt_lines = int(self.ch.cfg.get("vtt_lines", 2))
        groups = cc608.layout_text(text, vtt_chars, vtt_lines,
                                   self.ch.cfg.get("cc_layout") or "balanced", ascii_only,
                                   self.ch.cfg.get("sentences", True))
        for ca, cb, g in cc608.timed_groups(a, b, groups):
            if cues:
                ps, pe, pt = cues[-1]
                ca = max(ca, ps + 1800)          # at least one frame later
                if cb <= ca:
                    continue
                if pe > ca:
                    cues[-1] = (ps, ca, pt)
            cues.append((ca, cb, "\n".join(g)))

    def place(self, a, b, text, regions):
        """(a, b, piece) for the text's sentence pieces, spread over the speech
        regions inside a..b in proportion to their length (uniformly over
        a..b when there are none)."""
        if not self.ch.cfg.get("sentences", True):
            return [(a, b, text)]
        width = int(self.ch.cfg.get("vtt_chars", VTT_CHARS)) or 10 ** 6
        pieces = cc608.sentence_pieces(re.sub(r"\s+", " ", text).strip(), width,
                                       int(self.ch.cfg.get("vtt_lines", 2)))
        regs = [(max(a, s), min(b, e)) for s, e in regions if min(b, e) > max(a, s)] or [(a, b)]
        total = sum(e - s for s, e in regs)
        chars = sum(len(p) for p in pieces) or 1

        def at(frac):
            x = frac * total
            for s, e in regs:
                if x <= e - s:
                    return s + x
                x -= e - s
            return regs[-1][1]
        out, c = [], 0
        for p in pieces:
            s = at(c / chars)
            c += len(p)
            out.append((int(s), int(at(c / chars)), p))
        return out

    def sentence_flow(self, cues):
        """Main-track cues with an unfinished last sentence held back to be
        joined with the next chunk's text (so cues show whole sentences),
        while there is time; a held fragment is never held twice."""
        if not self.ch.cfg.get("sentences", True):
            return cues
        cues = list(cues)
        held, self.pending = self.pending, None
        if held:
            if cues:
                a, b, text = cues[0]
                joined = held[2] + " " + text.strip()
                cues[0:1] = self.place(held[0], b, joined, [])   # re-split the joined sentence(s)
            else:
                cues = [held[:3]]
            return cues
        if not cues or cc608.ends_sentence(cues[-1][2]):
            return cues
        a, b, text = cues[-1]
        anchor = self.anchor
        lead = (anchor[0] + (a - anchor[1]) / TICKS - time.monotonic()) if anchor else 0
        if lead < HOLD_MIN_LEAD:
            return cues
        self.pending = (a, b, text.strip(), self.chunk_no)
        return cues[:-1]

    def extra_tracks(self, audio, start, asr_lang, lang, task, ratio):
        """Comparison tracks (NLLB, Whisper, original) when Seamless is the
        main track: one /v1/multi request after the main one."""
        try:
            wanted = [t for t in ("nllb", "whisper") if t in hls.compare_mts(self.ch.cfg)]
            if not wanted:
                return   # only Seamless chosen: the main track is all there is
            r, el = self.asr_call(audio, start, asr_lang, task, wanted + ["original"])
        except Exception as e:  # noqa: BLE001 - only the comparison tracks miss the chunk
            self.drop(f"compare_{getattr(e, 'reason', type(e).__name__)}")
            return
        self.stats["compare_last_s"] = round(el, 2)
        if r.get("speech_ratio") is None:
            r["speech_ratio"] = ratio
        with self.vtt_lock:
            self.compare(audio, start, lang, task, "seamless", r)

    def seamless_track(self, audio, start, lang):
        """English (Seamless) cue for one chunk, from a parallel request."""
        try:
            r, el = self.asr_call(audio, start, lang, "translate", "seamless")
        except Exception as e:  # noqa: BLE001 - only this track misses the chunk
            self.drop(f"seamless_{getattr(e, 'reason', type(e).__name__)}")
            return
        self.stats["seamless_last_s"] = round(el, 2)
        guard = self.compare_guards.setdefault("seamless", asr.Guard())
        regions = [(start + int(s * TICKS), start + int(e * TICKS)) for s, e in asr.speech_regions(audio)]
        with self.vtt_lock:
            for s in r["segments"]:
                if (asr.seamless_check(s, r.get("speech_ratio"))
                        or guard.keep(s, r.get("speech_ratio"))) is None:
                    for a, b, t in self.place(start + int(s["start"] * TICKS), start + int(s["end"] * TICKS),
                                              s["text"], regions):
                        self.add_vtt("seamless", a, b, t)

    def compare(self, audio, start, lang, task, primary, r):
        """Extra subtitle tracks for comparing models (channel option
        "compare"): the source-language transcript and the other
        translation method. The main track has already been emitted."""
        ratio = r.get("speech_ratio")
        src_lang = lang or r.get("language") or "orig"
        if "tracks" in r:   # API v4: everything came back in one response
            for name, segs in r["tracks"].items():
                track = "orig" if name == "original" else name
                if name == primary:
                    continue
                guard = self.compare_guards.setdefault(track, asr.Guard())
                for s in segs:
                    if track == "orig" or guard.keep(s, ratio) is None:
                        self.add_vtt(track, start + int(s["start"] * TICKS), start + int(s["end"] * TICKS),
                                     s["text"], ascii_only=track != "orig")
            for name, err in r.get("errors", {}).items():
                self.drop(f"{name}_{err.split(':')[0][:20]}")
            return
        # Original language: the transcript the NLLB path translated from.
        for s in r["segments"]:
            if s.get("source_text") and not s.get("drop"):
                self.add_vtt("orig", start + int(s["start"] * TICKS), start + int(s["end"] * TICKS),
                             s["source_text"], ascii_only=False)
        self.ch.track_names["orig"] = f"Original ({src_lang})"
        if task != "translate":
            return
        other = "whisper" if primary == "nllb" else "nllb"
        try:
            r2, el = self.asr_call(audio, start, lang, task, other)
        except Exception as e:  # noqa: BLE001 - only this track misses the chunk
            self.drop(f"compare_{type(e).__name__}")
            return
        self.stats["compare_last_s"] = round(el, 2)
        guard = self.compare_guards.setdefault(other, asr.Guard())
        kept = 0
        for s in r2["segments"]:
            why = guard.keep(s, r2.get("speech_ratio") if r2.get("speech_ratio") is not None else ratio)
            if why is None:
                kept += 1
                self.add_vtt(other, start + int(s["start"] * TICKS), start + int(s["end"] * TICKS), s["text"])
        log.info("%s: compare %s: %d of %d segments kept (%.1fs)", self.ch.id, other, kept,
                 len(r2["segments"]), el)
        # Seamless last: it is the slowest, and only its own track waits.
        if "seamless" in getattr(backend, "mts", ()):
            try:
                r3, el3 = self.asr_call(audio, start, lang, task, "seamless")
            except Exception as e:  # noqa: BLE001 - only this track misses the chunk
                self.drop(f"seamless_{type(e).__name__}")
                return
            self.stats["seamless_last_s"] = round(el3, 2)
            guard = self.compare_guards.setdefault("seamless", asr.Guard())
            for s in r3["segments"]:
                if guard.keep(s, r3.get("speech_ratio") if r3.get("speech_ratio") is not None else ratio) is None:
                    self.add_vtt("seamless", start + int(s["start"] * TICKS), start + int(s["end"] * TICKS),
                                 s["text"])
        self.ch.track_names[other] = "English (Whisper)" if other == "whisper" else "English (NLLB)"

    def source_cue(self, pts, text):
        """The source's own caption (CC1) changed at pts: a WebVTT cue for it
        (ended by the next one, or by an erase)."""
        cues = self.ch.vtt_tracks["main"]
        if not text:
            if cues and cues[-1][0] < pts < cues[-1][1]:
                cues[-1] = (cues[-1][0], pts, cues[-1][2])
            return
        self.add_vtt("main", pts, pts + 6 * TICKS, text, ascii_only=False)

    def emit(self, a, b, text, log_it=True):
        with self.lock:
            self.encoder.split_segment(a, b, text, self.ch.cfg.get("cc_position") or "bottom",
                                       self.ch.cfg.get("cc_center") or CC_CENTER,
                                       self.ch.cfg.get("cc_layout") or "balanced")
        self.add_vtt("main", a, b, text)
        if log_it:
            self.stats["captions"] += 1
            self.ch.recent.append({"pts": round(a / TICKS, 1), "text": text.strip(),
                                   "time": time.strftime("%H:%M:%S", time.gmtime())})
            log.info("%s: [%.1f-%.1f] %s", self.ch.id, a / TICKS, b / TICKS, text.strip())

    # ----------------------------------------------------------- output ---

    def output_loop(self):
        out, ostreams = None, {}
        vtemplate = atemplate = None
        anchor = None   # (wall, ticks): release time = wall + (ts - ticks)/TICKS
        staging = []    # video packets waiting for a caption pair, decode order
        pending = []    # heap of (pts, seq) of staged packets without a pair
        source_cc = {}  # seq -> the source's own 608 pairs (CC auto)
        cc_decoder = cc608dec.Decoder()
        pairs = {}      # seq -> pair
        seq = 0
        REORDER = 16
        established = False  # lead has reached MIN_LEAD at least once
        self.ch.broadcaster.reset()
        sink = SessionSink(self)

        last_dts = {}

        def emit(kind, pkt, data=None):
            nonlocal out
            if out is None:
                out = av.open(sink, mode="w", format="mpegts",
                              options={"mpegts_flags": "resend_headers"})
                ostreams["v"] = out.add_stream_from_template(vtemplate)
                ostreams["a"] = out.add_stream_from_template(atemplate)
            if data is not None:
                np_ = av.Packet(data)
                np_.pts, np_.dts, np_.time_base = pkt.pts, pkt.dts, pkt.time_base
                np_.is_keyframe = pkt.is_keyframe
                pkt = np_
            # The muxer rejects non-increasing DTS (seen at ad-break
            # discontinuities); drop such packets rather than die.
            if pkt.dts is not None:
                if kind in last_dts and pkt.dts <= last_dts[kind]:
                    self.stats["ts_drops"] = self.stats.get("ts_drops", 0) + 1
                    return
                last_dts[kind] = pkt.dts
                if pkt.pts is not None and pkt.pts < pkt.dts:
                    pkt.pts = pkt.dts
            pkt.stream = ostreams[kind]
            try:
                out.mux(pkt)
            except av.error.FFmpegError as e:
                self.stats["mux_errors"] = self.stats.get("mux_errors", 0) + 1
                log.warning("%s: mux error (%s), packet dropped", self.ch.id, e)

        try:
            while not self.stop.is_set():
                try:
                    arrival, kind, pkt = self.outq.get(timeout=1)
                except queue.Empty:
                    continue
                if kind == "v" and vtemplate is None:
                    vtemplate = pkt.stream
                if kind == "a" and atemplate is None:
                    atemplate = pkt.stream
                if vtemplate is None or atemplate is None:
                    continue  # need both streams before the muxer can start
                ts = pkt.dts
                if anchor is None:
                    # Start at once, the first BURST seconds immediately: the
                    # HLS backlog provides the lead.
                    anchor = self.anchor = (arrival + self.start_delay - self.burst, ts)
                due = anchor[0] + (ts - anchor[1]) / TICKS
                lead = due - arrival
                # With a margin: right after a cold start the lead grows as the
                # backlog is read, and wobbles around MIN_LEAD on the way up;
                # re-anchoring there paused the output for nothing.
                established = established or lead >= MIN_LEAD + 5
                # Startup runs on the HLS backlog; once it should have built
                # up, keep at least MIN_LEAD so ASR has time to finish.
                if ((established or ts - anchor[1] > 30 * TICKS) and lead < MIN_LEAD
                        or lead > DELAY + 60):
                    log.info("%s: re-anchoring output clock (lead %.1fs)", self.ch.id, lead)
                    anchor = self.anchor = (arrival + DELAY, ts)
                    due = arrival + DELAY
                self.stats["lead_s"] = round(lead, 1)
                wait = due - time.monotonic()
                if wait > 0:
                    self.stop.wait(wait)

                if kind == "a":
                    emit("a", pkt)
                    continue

                # Video: assign caption pairs in presentation order.
                seq += 1
                staging.append((seq, pkt))
                heapq.heappush(pending, (pkt.pts, seq))
                if self.use_source_cc:   # their text goes to the WebVTT track too
                    source_cc[seq] = cc608dec.cc_pairs(bytes(pkt))
                while len(pending) > REORDER:
                    pts, s = heapq.heappop(pending)
                    with self.lock:
                        pairs[s] = self.encoder.next_frame(pts)
                    if s in source_cc:
                        text = cc_decoder.feed(source_cc.pop(s))
                        if text is not None:
                            self.source_cue(pts, text)
                while staging and staging[0][0] in pairs:
                    s, p = staging.pop(0)
                    pair, dtv = pairs.pop(s)
                    data = bytes(p)
                    self.watch_source_cc(data)
                    stripped = data if self.use_source_cc else cc608.strip_cc(data, self.stats.get("video_codec"))
                    if self.captionable and not self.use_source_cc and self.ch.cfg.get("eia", EIA_DEFAULT):
                        emit("v", p, cc608.insert_sei(stripped, cc608.sei_nal(pair, dtv)))
                    elif stripped is not data:   # the source's captions dropped (we generate ours)
                        emit("v", p, stripped)
                    else:
                        emit("v", p)
        finally:
            if out is not None:
                try:
                    out.close()
                except Exception:  # noqa: BLE001
                    pass


def segment_durations(url, headers):
    """Segment durations listed in an HLS media playlist, or [] (master
    playlist, not HLS, or unreachable)."""
    if ".m3u" not in url.split("?")[0]:
        return []
    try:
        h = {"User-Agent": headers.get("user_agent") or "Mozilla/5.0"}
        if headers.get("referer"):
            h["Referer"] = headers["referer"]
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=10) as r:
            text = r.read(2 * 1024 * 1024).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - no burst then
        return []
    return [float(d) for d in re.findall(r"#EXTINF:\s*([\d.]+)", text)]


def pick_variant(url, headers):
    """For an HLS master playlist, the URL of the best variant up to 1080p.

    Opening a master playlist makes libavformat fetch every variant; we only
    need one. Variants whose audio is a separate rendition are skipped,
    because the relay needs audio and video in one stream. Returns `url`
    unchanged for media playlists, non-HLS URLs and anything unexpected.
    """
    if ".m3u" not in url.split("?")[0]:
        return url
    try:
        h = {"User-Agent": headers.get("user_agent") or "Mozilla/5.0"}
        if headers.get("referer"):
            h["Referer"] = headers["referer"]
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=15) as r:
            final, text = r.geturl(), r.read(512 * 1024).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - let libavformat try the URL itself
        return url
    best, lines = None, text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("#EXT-X-STREAM-INF:") or i + 1 >= len(lines):
            continue
        attrs = dict(re.findall(r'([A-Z-]+)=("[^"]*"|[^,]*)', line))
        attrs = {k: v.strip('"') for k, v in attrs.items()}
        height = int(attrs.get("RESOLUTION", "0x0").split("x")[-1] or 0)
        if height > 1080 or "AUDIO" in attrs:
            continue
        if attrs.get("CODECS") and "mp4a" not in attrs["CODECS"]:
            continue
        key = (int(attrs.get("BANDWIDTH", 0) or 0), height)
        if best is None or key > best[0]:
            best = (key, urllib.parse.urljoin(final, lines[i + 1].strip()))
    return best[1] if best else url


def find_cut(audio):
    """Sample index to cut at: the middle of the first pause of PAUSE_MS
    after CHUNK_MIN, or the quietest point once the buffer reaches CHUNK_MAX."""
    win = SR // 50
    n = len(audio) // win
    energy = np.sqrt(np.mean(audio[:n * win].reshape(n, win) ** 2, axis=1))
    smooth = np.convolve(energy, np.ones(3) / 3, mode="same")
    lo = int(CHUNK_MIN * 50) - 1
    if n <= lo:
        return None
    quiet = smooth < max(10 ** (-50 / 20), 0.25 * float(np.median(smooth[:n])))
    need = max(1, PAUSE_MS // 20)
    run = 0
    for i in range(lo - need, n):
        run = run + 1 if quiet[i] else 0
        if run >= need and i - need // 2 >= lo:
            return (i - need // 2) * win
    if len(audio) < CHUNK_MAX * SR:
        return None
    smooth = np.convolve(energy, np.ones(10) / 10, mode="same")
    i = int(np.argmin(smooth[lo:n]))
    return (lo + i) * win


class Channel:
    """A configured channel; owns a Session while anyone is watching."""

    def __init__(self, cfg):
        self.cfg = cfg  # live entry from the config file; edits apply at once
        self.id = cfg["id"]
        self.broadcaster = Broadcaster()
        self.recent = collections.deque(maxlen=20)
        # WebVTT tracks for the HLS output: name -> deque of (start, end, text).
        # "main" is the default subtitle track; others exist with "compare".
        self.vtt_tracks = {"main": collections.deque(maxlen=300)}
        self.vtt_cues = self.vtt_tracks["main"]
        self.track_names = {"main": "English"}
        self.hls = None                                  # hls.Segmenter while HLS is requested
        self.burner = None                               # burn.Burner, video source in compare mode
        self.session = None
        self.idle_since = None
        self.lock = threading.Lock()

    name = property(lambda self: self.cfg["name"])
    number = property(lambda self: str(self.cfg.get("number") or ""))
    url = property(lambda self: self.cfg["url"])
    language = property(lambda self: self.cfg.get("language") or "auto")
    logo = property(lambda self: self.cfg.get("tvg", {}).get("tvg-logo", ""))
    # Segments back from the live edge to start, so the backlog covers the
    # delay; the default assumes 6 s segments.
    live_start_index = property(lambda self: int(self.cfg.get("live_start_index",
                                                                -int(DELAY // 6 + 1))))

    def urls(self):
        """This channel's URL, then working URLs of other feeds with the same tvg-id."""
        return cfg.feed_urls(self.cfg)

    def video_source(self):
        """Where the HLS video comes from: this channel's own stream, or with
        the Compare option the re-encode with the backends burned in."""
        if not self.cfg.get("burn", self.cfg.get("compare")):
            return self
        with self.lock:
            if self.burner is None:
                self.burner = burn.Burner(self, lambda: burn_tracks(
                    self.cfg, self.session and (self.language if self.language != "auto" else self.session.language)),
                    Broadcaster())
        return self.burner

    def subscribe(self):
        cq = queue.Queue(maxsize=4000)
        cq.waiting = True  # until a video keyframe starts
        with self.lock:
            with self.broadcaster.lock:
                self.broadcaster.clients.add(cq)
            self.idle_since = None
            if self.session is None:
                log.info("%s: starting session", self.id)
                self.session = Session(self)
                self.session.start()
        return cq

    def unsubscribe(self, cq):
        with self.lock:
            with self.broadcaster.lock:
                self.broadcaster.clients.discard(cq)
                if not self.broadcaster.clients:
                    self.idle_since = time.monotonic()

    def tick(self):
        """Called periodically: stop idle sessions, restart stalled ones."""
        with self.lock:
            s = self.session
            if s is None:
                return
            if self.idle_since and time.monotonic() - self.idle_since > GRACE:
                log.info("%s: no clients for %.0fs, stopping session", self.id, GRACE)
                s.stop.set()
                self.session = None
            elif (s.stats["input"] == "streaming"
                  and time.monotonic() - s.last_input > STALL
                  or not all(t.is_alive() for t in s.threads)):
                log.warning("%s: input stalled or a thread died, restarting session", self.id)
                s.stop.set()
                self.session = Session(self)
                self.session.start()

    def status(self):
        s = self.session
        return {"name": self.name, "number": self.number, "url": f"{PUBLIC_URL}/{self.id}.ts",
                "language": self.language, "clients": len(self.broadcaster.clients),
                "session": None if s is None else s.stats,
                "burned_compare": None if self.burner is None else self.burner.stats,
                "recent": list(self.recent)[-10:]}


cfg = None       # config.Config, loaded in main()
guide = epg.Guide()
checker = None   # health.Checker, started in main()
channels = {}    # id -> Channel, created on first use


def channel(cid, enabled_only=True):
    c = adhoc.channel_cfg(cid)   # stateless mode: the options are in the id
    if c is not None:
        with config.lock:
            if cid not in channels:
                channels[cid] = Channel(c)
            return channels[cid]
    with config.lock:
        c = cfg.get(cid)
        if c is None or (enabled_only and not c.get("enabled")):
            return channels.get(cid) if not enabled_only else None
        if cid not in channels:
            channels[cid] = Channel(c)
        channels[cid].cfg = c
        return channels[cid]
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html"), "rb") as f:
    UI = f.read()
_health = {"at": 0.0, "value": None}


def burn_tracks(c, lang=None):
    """(track, label, colour) burned in in a channel's compare mode, bottom
    band first. An English source has nothing to translate: its main track is
    Whisper's transcript, compared with Seamless."""
    style = {"nllb": ("NLLB", (255, 255, 255)), "whisper": ("WSP", (90, 230, 255)),
             "seamless": ("SML", (255, 225, 60))}
    chosen = hls.compare_mts(c)
    if lang == "en":
        return ([("main", *style["whisper"])] +
                ([("seamless", *style["seamless"])] if "seamless" in chosen else []))
    mt = hls.main_mt(c)
    return [("main" if t == mt else t, *style[t]) for t in chosen]


def asr_health():
    """The remote ASR service's /health, cached for a few seconds."""
    remote = getattr(backend, "primary", backend)
    if not isinstance(remote, asr.Remote):
        return None
    if time.monotonic() - _health["at"] > 5:
        v = {"url": remote.url}
        try:
            with urllib.request.urlopen(remote.url + "/health", timeout=1.5) as r:
                v.update(json.load(r))
        except Exception as e:  # noqa: BLE001 - shown on the status page
            v.update(ok=False, error=f"{type(e).__name__}: {e}")
        _health.update(at=time.monotonic(), value=v)
    return _health["value"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):
        log.info("http %s %s", self.client_address[0], fmt % args)

    def do_GET(self):
        path = self.path.split("?")[0].strip("/")
        if path.startswith("hls/"):
            return self.hls(path)
        if path in ("api/stream", "api/m3u"):
            return self.adhoc(path)
        if path == "stream.ts":  # pre-multi-channel URL
            enabled = cfg.enabled()
            path = (enabled[0]["id"] if enabled else "none") + ".ts"
        if path.endswith(".ts"):
            ch = channel(path[:-3])
            return self.stream(ch) if ch else self.reply(404, "text/plain", b"no such channel\n")
        if path == "playlist-hls.m3u":
            # Same channels and numbers as playlist.m3u, served as HLS with a
            # WebVTT English subtitle track (Channels DVR shows these centred).
            body = f'#EXTM3U url-tvg="{PUBLIC_URL}/epg.xml" x-tvg-url="{PUBLIC_URL}/epg.xml"\n'
            for c in cfg.enabled():
                tvg = c.get("tvg", {})
                logo = f' tvg-logo="{tvg["tvg-logo"]}"' if tvg.get("tvg-logo") else ""
                group = tvg.get("group-title") or "captions-dvr"
                body += (f'#EXTINF:-1 channel-id="{c["id"]}" tvg-id="{tvg.get("tvg-id") or c["id"]}" '
                         f'tvg-chno="{c.get("number") or ""}" tvg-name="{c["name"]}"{logo} '
                         f'group-title="{group}",{c["name"]}\n'
                         f"{PUBLIC_URL}/hls/{c['id']}/master.m3u8\n")
                if c.get("compare") and c.get("compare_channels") and c.get("number"):
                    # Channels DVR keeps one subtitle track per channel, so
                    # each comparison track gets its own sub-channel N.1, N.2...
                    for k, (track, label) in enumerate(hls.extra_tracks(c), 1):
                        body += (f'#EXTINF:-1 channel-id="{c["id"]}-{track}" tvg-id="{tvg.get("tvg-id") or c["id"]}" '
                                 f'tvg-chno="{c["number"]}.{k}" tvg-name="{c["name"]} [{label}]"{logo} '
                                 f'group-title="{group}",{c["name"]} [{label}]\n'
                                 f"{PUBLIC_URL}/hls/{c['id']}/master-{track}.m3u8\n")
            return self.reply(200, "audio/x-mpegurl", body.encode())
        if path == "epg.xml":
            with guide.lock:
                body = guide.xml
            return self.reply(200, "application/xml; charset=utf-8", body)
        if path in ("playlist.m3u", "playlist.m3u8"):
            body = f'#EXTM3U url-tvg="{PUBLIC_URL}/epg.xml" x-tvg-url="{PUBLIC_URL}/epg.xml"\n'
            for c in cfg.enabled():
                tvg = c.get("tvg", {})
                logo = f' tvg-logo="{tvg["tvg-logo"]}"' if tvg.get("tvg-logo") else ""
                group = tvg.get("group-title") or "captions-dvr"
                body += (f'#EXTINF:-1 channel-id="{c["id"]}" tvg-id="{tvg.get("tvg-id") or c["id"]}" '
                         f'tvg-chno="{c.get("number") or ""}" tvg-name="{c["name"]}"{logo} '
                         f'group-title="{group}",{c["name"]}\n'
                         f"{PUBLIC_URL}/{c['id']}.ts\n")
            return self.reply(200, "audio/x-mpegurl", body.encode())
        if path == "api/config":
            with config.lock:
                body = json.dumps({"sources": cfg.sources, "channels": cfg.channels, "settings": cfg.settings,
                                   "languages": languages.data,
                                   "health": {"running": checker.running, "progress": checker.progress},
                                   "epg": guide.matches, "epg_updated": guide.updated,
                                   "whisper_langs": sorted(asr.WHISPER_LANGS)}).encode()
            return self.reply(200, "application/json", body)
        if path in ("", "index.html"):
            return self.reply(200, "text/html; charset=utf-8", UI)
        if path == "status":
            body = json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                               "asr_backend": type(backend).__name__,
                               "asr_health": asr_health(),
                               "asr_last_error": getattr(backend, "last_error", None),
                               "channels": {c["id"]: channel(c["id"]).status() for c in cfg.enabled()},
                               "adhoc": {cid: c.status() for cid, c in list(channels.items())
                                         if c.cfg.get("adhoc") and c.session}},
                              indent=1).encode()
            return self.reply(200, "application/json", body)
        return self.reply(404, "text/plain", b"not found\n")

    def adhoc(self, path):
        """Stateless mode: /api/m3u rewrites a playlist to /api/stream URLs;
        /api/stream serves one source as HLS (the master playlist)."""
        raw = self.path.partition("?")[2]
        q = urllib.parse.parse_qs(raw)
        # Lenient about the source URL: "?urlhttps%3A..." (the "=" lost in
        # copy and paste), and an unencoded url=...?a=1&b=2 (everything up to
        # the next relay parameter).
        m = re.search(r"(?:^|&)url=?(https?(?::|%3A).*?)(?=&(?:mt|lang|pos|compare|chno_offset|referer|user_agent)=|$)",
                      raw, re.I)
        if m:
            q["url"] = [urllib.parse.unquote(m.group(1))]
        try:
            if path == "api/m3u":
                body = adhoc.rewrite_m3u(adhoc.fetch_m3u((q.get("url") or [""])[0]), PUBLIC_URL, q)
                return self.reply(200, "audio/x-mpegurl", body.encode())
            o, mt = adhoc.options(q)
            cid = adhoc.token(o)
            ch = channel(cid)
            track = adhoc.want(ch.cfg, mt or hls.main_mt({}))
            if ch.session is None:
                active = sum(1 for c in list(channels.values()) if c.cfg.get("adhoc") and c.session)
                if active >= adhoc.MAX_SESSIONS:
                    raise adhoc.Refused(f"{active} streams already running (MAX_ADHOC_SESSIONS)", 503)
                adhoc.probe(ch.url, ch.cfg.get("headers") or {})
                log.info("ad-hoc stream %s: %s", cid[:20], o)
        except adhoc.Refused as e:
            log.info("ad-hoc refused (%d): %s", e.code, e)
            return self.reply(e.code, "text/plain", f"{e}\n".encode())
        return self.hls(f"hls/{cid}/master.m3u8" if track == "main" else f"hls/{cid}/master-{track}.m3u8")

    def hls(self, path):
        _, cid, name = (path.split("/") + ["", ""])[:3]
        ch = channel(cid)
        if ch is None:
            return self.reply(404, "text/plain", b"no such channel\n")
        if ch.cfg.get("adhoc"):
            # A backend variant's URL (e.g. after a relay restart): register
            # the backend and map it to the track that carries it.
            m = re.match(r"(master-|subs-|\d+-)(seamless|nllb|whisper)(\..*)", name)
            if m:
                track = adhoc.want(ch.cfg, m.group(2))
                name = (m.group(1) + track + m.group(3)) if track != "main" else \
                    {"master-": "master", "subs-": "subs"}.get(m.group(1), m.group(1)[:-1]) + m.group(3)
        if ch.hls is None:
            ch.hls = hls.Segmenter(ch)
        seg = ch.hls
        seg.touch()
        base = f"{PUBLIC_URL}/hls/{cid}"
        if name == "master.m3u8":
            return self.reply(200, "application/vnd.apple.mpegurl", seg.master(base).encode())
        if name.startswith("master-") and name.endswith(".m3u8"):   # one-track comparison channel
            return self.reply(200, "application/vnd.apple.mpegurl", seg.master(base, name[7:-5]).encode())
        if name.startswith("subs-") and name.endswith(".m3u8"):
            return self.reply(200, "application/vnd.apple.mpegurl",
                              seg.playlist(base, "subs", name[5:-5]).encode())
        if name in ("video.m3u8", "subs.m3u8"):
            # First request of a new session: wait for a segment, but never
            # hold the request long (a DVR timed out on a 30 s wait; with
            # BURST the first segment is typically there after 7-11 s).
            for _ in range(int(COLD_WAIT * 4)):
                if seg.segments:
                    break
                time.sleep(0.25)
            return self.reply(200, "application/vnd.apple.mpegurl",
                              seg.playlist(base, "video" if name == "video.m3u8" else "subs").encode())
        num, _, ext = name.partition(".")
        if num.isdigit() and ext == "ts":
            s = seg.segment(int(num))
            return self.reply(200, "video/mp2t", s[3]) if s else self.reply(404, "text/plain", b"gone\n")
        num, _, track = num.partition("-")
        if num.isdigit() and ext == "vtt":
            v = seg.vtt(int(num), track or "main")
            return self.reply(200, "text/vtt", v.encode()) if v else self.reply(404, "text/plain", b"gone\n")
        return self.reply(404, "text/plain", b"not found\n")

    def do_HEAD(self):
        """Headers only, for players and DVRs that probe URLs before GET."""
        path = self.path.split("?")[0].strip("/")
        if path == "api/stream" or path.endswith(".m3u8") or path.endswith(".m3u"):
            ctype = "audio/x-mpegurl" if path.endswith(".m3u") else "application/vnd.apple.mpegurl"
        elif path == "api/m3u":
            ctype = "audio/x-mpegurl"
        elif path.endswith(".ts"):
            ctype = "video/mp2t"
        elif path.endswith(".vtt"):
            ctype = "text/vtt"
        elif path in ("", "index.html"):
            ctype = "text/html; charset=utf-8"
        elif path in ("epg.xml",):
            ctype = "application/xml; charset=utf-8"
        else:
            ctype = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()

    def do_POST(self):
        """Config API used by the web page. LAN only; no authentication."""
        path = self.path.split("?")[0].strip("/")
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if path == "api/sources":
                src, added = cfg.add_source(data["spec"], data.get("name"))
                return self.json(200, {"source": src, "added": added})
            if path == "api/languages/forget":
                languages.forget(data["url"])
                return self.json(200, {"ok": True})
            if path == "api/settings":
                return self.json(200, cfg.update_settings(data))
            if path == "api/check":
                checker.trigger()
                return self.json(200, {"started": True})
            if path == "api/sources/remove":
                cfg.remove_source(data["id"])
                return self.json(200, {"ok": True})
            if path.startswith("api/channels/"):
                c = cfg.update_channel(path[len("api/channels/"):], data)
                if "enabled" in data or "translate_guide" in data:
                    guide.wake.set()  # rebuild the guide for the new channel set
                return self.json(200, c)
        except KeyError as e:
            return self.json(404, {"error": f"unknown {e}"})
        except Exception as e:  # noqa: BLE001 - shown in the page
            return self.json(400, {"error": f"{type(e).__name__}: {e}"})
        return self.json(404, {"error": "not found"})

    def json(self, code, obj):
        return self.reply(code, "application/json", json.dumps(obj).encode())

    def reply(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if "mpegurl" in ctype or ctype.startswith("application/json"):
            self.send_header("Cache-Control", "no-cache")  # live playlists and status
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream(self, ch):
        self.send_response(200)
        self.send_header("Content-Type", "video/mp2t")
        self.end_headers()
        cq = ch.subscribe()
        try:
            while True:
                chunk = cq.get(timeout=60)
                if chunk is None:
                    break
                self.wfile.write(chunk)
        except (OSError, queue.Empty):
            pass
        finally:
            ch.unsubscribe(cq)


def housekeeping():
    while True:
        time.sleep(5)
        for ch in list(channels.values()):
            ch.tick()


def main():
    global cfg
    if not os.path.exists(CHANNELS_FILE) and os.path.exists(SEED_FILE):
        shutil.copy(SEED_FILE, CHANNELS_FILE)  # first run: start from the bundled channels
    cfg = config.Config(CHANNELS_FILE)
    cfg.save()  # writes the current format
    threading.Thread(target=housekeeping, name="housekeeping", daemon=True).start()
    hls.BACKEND = backend
    hls.COUNTRIES = lambda c: config.channel_countries(c, cfg.sources)
    hls.COUNTRY_LANG = epg.guide_language
    hls.SETTINGS = cfg.settings
    slate.CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(CHANNELS_FILE)), "slate")
    threading.Thread(target=slate.prepare, name="slate", daemon=True).start()
    if isinstance(backend, asr.Remote):
        backend.api()   # learn the server's tracks (e.g. seamless) before serving playlists
    global checker
    checker = health.Checker(cfg, pick_variant, config)
    threading.Thread(target=checker.loop, name="health", daemon=True).start()
    if any(c.get("major") and "health" not in c for c in cfg.channels):
        checker.trigger()  # first run: check what hasn't been checked
    threading.Thread(target=guide.loop, name="epg", daemon=True, args=(
        lambda: [(c, config.channel_countries(c, cfg.sources)) for c in cfg.enabled()],)).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    log.info("serving %d enabled channels on :%d, playlist %s/playlist.m3u",
             len(cfg.enabled()), PORT, PUBLIC_URL)
    srv.serve_forever()


if __name__ == "__main__":
    main()
