"""HLS output with a WebVTT subtitle rendition.

Channels DVR passes WebVTT subtitles through to its clients and centres
them, which CEA-608/708 placement could not achieve on tvOS.

A Segmenter subscribes to a channel like any TS client, cuts the stream at
video keyframes into ~6 s segments and keeps the last few in memory. The
subtitle playlist mirrors the video segments; each .vtt segment holds the
cues overlapping it, timed on the same 90 kHz clock as the video
(X-TIMESTAMP-MAP MPEGTS:0 = LOCAL 0).
"""
import collections
import logging
import os
import threading
import time

import slate

log = logging.getLogger("hls")
PIECE = 2.0   # segment length (cut at any video frame, see Segmenter.run)

TARGET = 6.0
KEEP = 15   # segments in the playlist (~30 s)
IDLE = 60.0
VIDEO_PID = 0x100
# Optional blank cue (CUE_BLANK, its text; off by default) 0.1 s long after
# a caption that 3 s or more of silence follows, for players that keep the
# last caption up until another replaces it. Channels DVR drops cues of only
# (no-break) spaces, and the Channels app on tvOS drew a zero-width space as
# an empty caption box that stayed up just the same.
BLANK, BLANK_GAP = 9000, 270000
BLANK_TEXT = os.environ.get("CUE_BLANK", "")
TABLE_PIDS = (0x0000, 0x1000)   # PAT, and the PMT where libavformat's muxer puts it
# libavformat's mpegts muxer adds max_delay (0.7 s) to every PTS it writes;
# cue times are on the relay's clock, before that offset.
MUX_DELAY = 63000
# WebVTT cue line per caption position setting.
VTT_LINE = {"bottom": "85%", "middle": "65%", "top": "10%"}
EIA_DEFAULT = os.environ.get("EIA", "0") == "1"   # as in relay.py
BACKEND = None   # the relay's ASR backend (set by relay.main); its .mts says which tracks exist
COUNTRIES = None      # set by relay.main: channel cfg -> countries
COUNTRY_LANG = None   # set by relay.main: (cfg, countries) -> likely spoken language
LABELS = {"nllb": "NLLB", "whisper": "Whisper", "seamless": "Seamless"}
LANG_NAMES = {"ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fa": "Persian",
              "fr": "French", "he": "Hebrew", "ja": "Japanese", "ko": "Korean", "lv": "Latvian",
              "ru": "Russian", "tr": "Turkish", "uk": "Ukrainian", "zh": "Chinese"}
SETTINGS = {}         # set by relay.main: the relay-wide settings (Config.settings)


def main_mt(cfg):
    """The channel's main translation method: its "mt", else Seamless when the
    ASR server has it, else NLLB."""
    return cfg.get("mt") or ("seamless" if "seamless" in getattr(BACKEND, "mts", ()) else "nllb")


def compare_mts(cfg):
    """Translation methods a "compare" channel shows (the channel's
    compare_mts, else the relay's Compare defaults, else all the ASR server
    offers), in band order."""
    have = ["nllb", "whisper"] + (["seamless"] if "seamless" in getattr(BACKEND, "mts", ()) else [])
    return [m for m in have if m in (cfg.get("compare_mts") or SETTINGS.get("compare_mts") or have)]


def extra_tracks(cfg):
    """(track, label) of the comparison tracks besides "main" (channel option
    "compare"): the other chosen translation methods, then the original language."""
    mt = main_mt(cfg)
    return [(t, LABELS[t]) for t in compare_mts(cfg) if t != mt] + [("orig", "Original")]


def pes_pts(pkt):
    """PTS (90 kHz) from a TS packet that starts a PES, or None."""
    off = 4 + (pkt[4] + 1 if pkt[3] & 0x20 else 0)
    p = pkt[off:]
    if len(p) < 14 or p[:3] != b"\x00\x00\x01" or not p[7] & 0x80:
        return None
    return (((p[9] >> 1) & 7) << 30 | p[10] << 22 | (p[11] >> 1) << 15 | p[12] << 7 | p[13] >> 1)


def head(tables):
    """PAT then PMT, from {pid: packet}."""
    return b"".join(tables[p] for p in TABLE_PIDS if p in tables)


def fmt(t):
    t = max(0.0, t)
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


class Segmenter:
    def __init__(self, channel):
        self.ch = channel
        self.segments = collections.deque(maxlen=KEEP)   # (seq, start_pts, dur, bytes)
        # Sequence numbers must keep increasing across relay and session
        # restarts, or a client that was already playing sees an "older"
        # playlist and loops. Start from the wall clock (segments are >= PIECE s
        # long, so an earlier session can't have got this far).
        self.seq = max(int(time.time() // PIECE), getattr(channel, "hls_next_seq", 0))
        self.disc_seq = int(time.time())     # EXT-X-DISCONTINUITY-SEQUENCE base
        self.disc = set()                    # seq numbers that follow a discontinuity
        self.last_request = time.monotonic()
        self.lock = threading.Lock()         # segments/seq: the slate thread appends too
        # Nothing to play yet (no session, or not output yet): serve the
        # slate card until the first real segment.
        s = channel.session
        # (A tuner of a format not seen before: run_slate waits for the
        # session to read it, then picks or renders a matching card.)
        self.slate_on = s is None or getattr(s, "anchor", None) is None
        # Video comes from the channel (passthrough) or, in compare mode,
        # from its burn.Burner; run() follows the channel's setting.
        self.src = channel.video_source()
        self.cq = self.src.subscribe()
        self.thread = threading.Thread(target=self.run, name=f"{channel.id}-hls", daemon=True)
        self.thread.start()
        if self.slate_on:
            threading.Thread(target=self.run_slate, name=f"{channel.id}-slate", daemon=True).start()

    def run_slate(self):
        """Slate segments: two at once, then one per segment length, until the
        first real segment (run() turns slate_on off) or the slate runs out."""
        # A format seen before: its card at once. A new one: wait (up to 5 s)
        # for the session to open the source, so the card is in its codec and
        # size (an H.264 card before MPEG-2 can show green), else take the
        # closest cached one.
        segs = slate.get(self.ch) if slate.fmt_of(self.ch) else None
        for _ in range(20):
            if segs or getattr(self.ch, "source_format", None):
                break
            time.sleep(0.25)
        if not segs and getattr(self.ch, "source_format", None):
            fmt = slate.fmt_of(self.ch)   # exactly its format; quick: a few seconds now, the rest later
            segs = slate._cached(slate._key(fmt)) or slate.quick(fmt)
        if not segs:
            segs = slate.get(self.ch)
        if not segs:
            with self.lock:
                self.slate_on = False
            return
        t0 = time.monotonic()
        k = 0
        while True:
            if k >= len(segs):
                # A quick card ran out: go on with the full one, in the same
                # format (same picture and timestamps), once it's rendered.
                fmt, more = slate.fmt_of(self.ch), None
                for _ in range(int(2 * slate.SEG / 0.25)):
                    more = slate._cached(slate._key(fmt)) if fmt else None
                    if more or not self.slate_on:
                        break
                    time.sleep(0.25)
                if not more or len(more) <= k:
                    return
                segs = more
            dur, data, pts = segs[k]
            with self.lock:
                if not self.slate_on:
                    return
                self.segments.append((self.seq, pts, dur, data))
                self.seq += 1
                self.ch.hls_next_seq = self.seq
            if k >= 1:
                time.sleep(max(0.0, t0 + (k * dur) - time.monotonic()))
            k += 1

    def run(self):
        # Segments are PIECE-second pieces cut at any video frame (they decode
        # in sequence), not whole keyframe intervals: published as soon as
        # they're complete, so a player at the live edge never waits for a
        # 6-10 s interval to close (that stalled it right after a cold start).
        # Only the first segment after a discontinuity starts at a keyframe.
        buf, start, rest, last = bytearray(), None, b"", 0
        # The latest PAT and PMT, put at the start of every segment: a
        # segment cut at a frame otherwise has them only ~0.1 s in, and
        # Channels DVR drops what comes before them after a discontinuity,
        # keyframe included (Android then smeared the next frames over the
        # slate card).
        tables = {}
        while time.monotonic() - self.last_request < IDLE:
            want = self.ch.video_source()
            if want is not self.src:
                # Compare mode switched: new encoding from the next keyframe,
                # marked as a discontinuity (sequence numbers keep going).
                cq = want.subscribe()
                self.src.unsubscribe(self.cq)
                self.src, self.cq = want, cq
                if start is not None and len(buf) > 188 * 100:
                    self.append(start, max(0.1, (last - start) / 90000), head(tables) + bytes(buf))
                with self.lock:
                    self.disc.add(self.seq)
                buf, start, rest = bytearray(), None, b""
            try:
                chunk = self.cq.get(timeout=1)
            except Exception:  # noqa: BLE001 - queue.Empty: keep waiting
                continue
            if chunk is None:
                # Dropped as a slow client (the queue overflowed): bytes were
                # lost, so resubscribe and carry on from the next keyframe as
                # a discontinuity. Exiting would empty the playlist and make
                # the next segmenter's sequence jump, which resets the DVR.
                log.warning("%s: segmenter fell behind, resuming at the next keyframe", self.ch.id)
                self.ch.segmenter_resumes = getattr(self.ch, "segmenter_resumes", 0) + 1
                self.cq = self.src.subscribe()
                with self.lock:
                    self.disc.add(self.seq)
                buf, start, rest = bytearray(), None, b""
                continue
            data = rest + chunk
            n = len(data) // 188 * 188
            rest = data[n:]
            for i in range(0, n, 188):
                pkt = data[i:i + 188]
                video_start = (pkt[0] == 0x47 and pkt[1] & 0x40
                               and ((pkt[1] & 0x1F) << 8 | pkt[2]) == VIDEO_PID)
                if video_start:
                    pts = pes_pts(pkt)
                    key = pkt[3] & 0x20 and pkt[4] and pkt[5] & 0x40
                    if pts is not None:
                        # (B-frames of an open GOP, as in broadcast MPEG-2, run
                        # up to a few frames before the I-frame: not a jump.)
                        if start is not None and not -2 * 90000 <= pts - start < 60 * 90000:
                            # Timestamps jumped (session restarted): close the
                            # segment here and mark the next one discontinuous.
                            if len(buf) > 188 * 100:
                                self.append(start, max(0.1, (last - start) / 90000), head(tables) + bytes(buf))
                            with self.lock:
                                self.disc.add(self.seq)
                            start, buf = None, bytearray()
                        if start is None:
                            if not key:
                                continue   # a new segment run starts at a keyframe
                            start, buf = pts, bytearray()
                        elif pts - start >= PIECE * 90000:
                            self.append(start, (pts - start) / 90000, head(tables) + bytes(buf))
                            start, buf = pts, bytearray()
                        last = pts
                elif ((pkt[1] & 0x1F) << 8 | pkt[2]) in TABLE_PIDS:
                    tables[(pkt[1] & 0x1F) << 8 | pkt[2]] = pkt
                if start is not None:
                    buf += pkt
        self.src.unsubscribe(self.cq)
        self.ch.hls = None

    def append(self, start, dur, data):
        """A real segment; the first one after the slate starts a discontinuity."""
        with self.lock:
            if self.slate_on:
                self.slate_on = False
                self.disc.add(self.seq)
            self.segments.append((self.seq, start, dur, data))
            self.seq += 1
            self.ch.hls_next_seq = self.seq   # a later segmenter continues from here

    def touch(self):
        self.last_request = time.monotonic()

    def master(self, base, only=None):
        """only: serve just that subtitle track, as the default. Channels DVR
        keeps only one subtitle rendition, so comparison channels each get
        their own master with a different track."""
        s = self.ch.session
        # Declare CC1 when we embed 608, or when the source's own captions
        # are passed through (e.g. US news channels in "CC: auto").
        eia = self.ch.cfg.get("eia", EIA_DEFAULT) or bool(s and s.use_source_cc)
        cc = ('#EXT-X-MEDIA:TYPE=CLOSED-CAPTIONS,GROUP-ID="cc",NAME="English CC",LANGUAGE="en",'
              'INSTREAM-ID="CC1",DEFAULT=NO,AUTOSELECT=NO\n') if eia else ""
        subs = ""
        src = self.ch.cfg.get("language") or "auto"
        names = {"main": "English" if len(self.tracks()) == 1 else
                 f"English ({LABELS[main_mt(self.ch.cfg)]})",
                 "whisper": "English (Whisper)", "nllb": "English (NLLB)",
                 "orig": "Original",
                 "seamless": "English (Seamless)"}
        detected = getattr(self.ch.session, "language", None) if self.ch.session else None
        if src == "auto" and not detected and COUNTRIES:
            detected = COUNTRY_LANG(self.ch.cfg, COUNTRIES(self.ch.cfg))
        orig_lang = src if src != "auto" else (detected or "und")
        # Distinct language tags: players and the DVR may merge renditions
        # that share one (e.g. three "en" tracks shown as one).
        tags = {"main": "en", "whisper": "en-US", "nllb": "en-GB", "seamless": "en-AU", "orig": orig_lang}
        for track in ([only] if only else self.tracks()):
            name = names.get(track, track)
            lang = tags.get(track, "und")
            uri = f"{base}/subs.m3u8" if track == "main" else f"{base}/subs-{track}.m3u8"
            first = track == (only or "main")
            subs += (f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="{name}",LANGUAGE="{lang}",'
                     f'DEFAULT={"YES" if first else "NO"},AUTOSELECT={"YES" if first else "NO"},'
                     f'FORCED=NO,URI="{uri}"\n')
        # No EXT-X-INDEPENDENT-SEGMENTS: a cold start's first pieces don't
        # begin with a keyframe.
        return ("#EXTM3U\n#EXT-X-VERSION:6\n" + subs + cc +
                '#EXT-X-STREAM-INF:BANDWIDTH=10000000,' +
                # CODECS only for H.264 + AAC; omitted for e.g. broadcast MPEG-2 + AC-3.
                ('CODECS="avc1.640028,mp4a.40.2",' if self.video_codec() == "h264" else "") +
                'SUBTITLES="subs",CLOSED-CAPTIONS=' + ('"cc"' if eia else "NONE") + "\n"
                f"{base}/video.m3u8\n")

    def video_codec(self):
        """The source's video codec: from the running session, else as seen on
        an earlier tune, else H.264 for HLS sources; None if unknown."""
        s = self.ch.session
        if s and s.stats.get("video_codec"):
            return s.stats["video_codec"]
        fmt = slate.fmt_of(self.ch)
        if fmt:
            return "h264" if fmt.get("vcodec", "h264") == "h264" else fmt["vcodec"]
        return "h264" if ".m3u" in (self.ch.cfg.get("url") or "").split("?")[0] else None

    def tracks(self):
        """Subtitle tracks to offer: main, plus comparison tracks when enabled.
        A new comparison model only needs to fill ch.vtt_tracks[<name>]."""
        if not self.ch.cfg.get("compare", False):
            return ["main"]
        names = ["main"] + [t for t, _ in extra_tracks(self.ch.cfg)]
        return names + [t for t in self.ch.vtt_tracks if t not in names]

    def playlist(self, base, kind, track="main"):
        segs = list(self.segments)
        first = segs[0][0] if segs else self.seq
        # Discontinuities that have slid out of the window still count.
        gone = sum(1 for d in self.disc if d <= first)
        out = ["#EXTM3U", "#EXT-X-VERSION:6", f"#EXT-X-TARGETDURATION:{int(TARGET * 2)}",
               f"#EXT-X-MEDIA-SEQUENCE:{first}",
               f"#EXT-X-DISCONTINUITY-SEQUENCE:{self.disc_seq + gone}"]
        for seq, _, dur, _ in segs:
            if seq in self.disc and seq != first:
                out.append("#EXT-X-DISCONTINUITY")
            if kind == "video":
                name = f"{seq}.ts"
            else:
                name = f"{seq}.vtt" if track == "main" else f"{seq}-{track}.vtt"
            out += [f"#EXTINF:{dur:.3f},", f"{base}/{name}"]
        return "\n".join(out) + "\n"

    def segment(self, seq):
        return next((s for s in self.segments if s[0] == seq), None)

    def vtt(self, seq, track="main"):
        s = self.segment(seq)
        if s is None:
            return None
        _, start, dur, _ = s
        # Same timing format as the 8599 test that Channels accepted:
        # MPEGTS:0 = LOCAL 0 and cue times on the (33-bit) PTS clock.
        lines = ["WEBVTT", "X-TIMESTAMP-MAP=MPEGTS:0,LOCAL:00:00:00.000", ""]
        line = VTT_LINE.get(self.ch.cfg.get("cc_position") or "bottom", "85%")
        cues = sorted(self.ch.vtt_tracks.get(track, ()))
        # A blank cue where a cue ends with nothing right after it: some players (Channels on iPhone
        # and Apple TV, through the DVR) keep showing the last cue until
        # another one replaces it, so a caption stayed up through silence or
        # an ASR outage.
        timed = []
        for i, (cs, ce, text) in enumerate(cues):
            timed.append((cs, ce, text))
            nxt = cues[i + 1][0] if i + 1 < len(cues) else None
            if BLANK_TEXT and (nxt is None or nxt - ce >= BLANK_GAP):
                timed.append((ce, ce + BLANK if nxt is None else min(ce + BLANK, nxt - 1800), BLANK_TEXT))
        base = start / 90000
        out = []
        for cs, ce, text in timed:
            a = wrap_diff(cs + MUX_DELAY, start) / 90000
            b = wrap_diff(ce + MUX_DELAY, start) / 90000
            # Each cue only in the segment where it starts (it may run past
            # the segment end): repeating it in the next segment makes some
            # players (Fire TV) draw it twice.
            if 0 <= a < dur:
                out.append((a, b, text))
        for a, b, text in out:
            lines += [f"{fmt(base + a)} --> {fmt(base + b)} align:center line:{line}", text, ""]
        return "\n".join(lines) + "\n"


def wrap_diff(t, ref):
    """t - ref in 90 kHz ticks, allowing for the 33-bit PTS wrap."""
    d = (t - ref) % (1 << 33)
    return d - (1 << 33) if d >= 1 << 32 else d
