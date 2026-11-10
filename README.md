# captions-dvr

Live TV (North Korean, Middle Eastern, or any iptv-org / M3U source) with
machine-translated English subtitles, for Channels DVR. The relay pulls each
channel's stream, translates the speech to English with SeamlessM4T v2 on
the Mac (English channels: Whisper transcription), and serves it to Channels
as **HLS with a WebVTT English subtitle track**, which Channels passes through to its clients
and keeps in recordings. Optionally (per channel "EIA CC", or `EIA=1`
for all) the same captions are also embedded in the video as CEA-608 and
CEA-708 (A/53 SEI); Channels doesn't need them. The video is never re-encoded.

```
source ──► relay ─┬─ video/audio ── delay ~20 s ── (+ 608/708 SEI) ──► HLS /hls/<id>/ (TS segments + WebVTT)
                  └─ audio 16 kHz ─► ASR backend (Mac: Seamless | mlx-whisper + NLLB; NVIDIA; none)
```

## Endpoints

| Path | What |
|---|---|
| `/playlist.m3u` | M3U for a Channels DVR Custom Channel, Stream Format **HLS**: each channel with an English WebVTT subtitle track (`/playlist-hls.m3u` is the same) |
| `/hls/<id>/master.m3u8` | One channel as HLS (6 s TS segments, `subs.m3u8` WebVTT rendition) |
| `/epg.xml` | XMLTV guide for the enabled channels (translated to English with `EPG_TRANSLATE=1`) |
| `/` | Web page: live status, sources and channel picker, per-channel settings |
| `/status` | JSON: per-channel session state, ASR speed, dropped-segment counts, recent captions |
| `/metrics` | Prometheus text format (Netdata, Prometheus): per-channel ASR time, lead, captions, drops, worker CPU/RAM; the ASR server has its own (docs/NVIDIA.md) |

Channels run on demand: the first client starts the pull and ASR, and the
session stops `GRACE` seconds after the last client leaves. Each running
channel has its own worker process (so channels use separate CPU cores); the
main process serves the web page and playlists, and queues every channel's
ASR requests, nearest deadline first. A session starts
from the HLS backlog, so clients get data straight away. Captions for the first
~15 s after a tune may be missing.

## Channels

`channels.json` (`CHANNELS_FILE`) holds the sources, channels and settings;
the web page writes it, and a new install starts with none. A hand-written
list of channels works too (`examples/channels.example.json`):

```json
[{"id": "livenow", "number": 8501, "name": "LiveNOW from FOX", "language": "en",
  "source": "https://fox-foxnewsnow-vizio.amagi.tv/playlist.m3u8"},
 {"id": "kctv", "number": 8502, "name": "KCTV", "language": "ko",
  "source": "https://live.koryocdn.org/kctv/index.m3u8"}]
```

Optional per channel: `logo`, and `live_start_index` (how many segments back
from the live edge to start, default `-(DELAY/6+1)`, which assumes 6 s segments).

Translation (`mt`, set in the web page): `seamless` is the default when the
Mac offers it (one GPU request per chunk, ~1.6 s for 6-12 s of audio, so a
Mac M4 carries roughly 5-6 channels); `nllb` and `whisper` remain. Seamless
has no confidence scores, so its cues are also dropped when VAD finds under
25% speech, when a phrase loops, or when there are more words than the speech
could hold; repeated sentences within a cue are collapsed. "Auto" language
channels assume the country's language (English ones are transcribed by
Whisper instead).

Comparing translation backends is opt-in per channel ("Compare" in the web
page, `compare`). The channel itself (same number) then switches to
re-encoded video with the chosen backends burned in, one labelled colour band
each: NLLB white at the
bottom, Whisper cyan, Seamless yellow above. Its WebVTT track stays. The
switch is a discontinuity in the HLS stream, so players may need a re-tune.
The backends and the resolution (480/720/1080p, default 1080p) are relay-wide
"Compare" settings in the web page (`settings` in `channels.json`; a channel
may override them with its own `compare_mts` / `burn_height`).
Encoding runs only while the channel is watched: libx264 (`BURN_ENCODER`,
`BURN_PRESET`); on the test host's
i5-11400 1080p takes about 1.5 cores, 360p about 0.8 (mostly decoding the
source), ~3 Mbit/s at 1080p. Extra backends cost Mac time: with all three the
Mac carries about one channel. With `compare_channels` the other backends and
the original-language transcript are also offered as plain subtitle
sub-channels N.1, N.2... (Channels DVR passes only the default rendition).

Subtitles follow sentences (per channel "sentences", default on): each
sentence starts a new cue, short ones share a cue when they fit, long ones
split at commas, and a sentence left unfinished at the end of a chunk is
joined to the next chunk's text. Seamless gives no timestamps, so its text is
spread over the chunk's speech (frame energy) by length. Every cue stays up
at least 1.2 s and long enough to read at 17 characters a second.

### ATSC 3.0 (HEVC, AC-4)

HEVC video passes through like any other (start cards are HEVC too). AC-4
audio can't be decoded by the ffmpeg in PyAV, so such sources need an
external ffmpeg binary with an AC-4 decoder: set its path (inside the relay)
with `AC4_FFMPEG` or "audio ffmpeg" in the web page. Such a source is then
read through it: video copied, audio converted to stereo AAC, for both the
DVR and ASR. Without it these channels fail with a clear error. No such
binary is shipped with captions-dvr.

## Stateless mode: captions for the DVR's own channels

The relay doesn't have to manage channels. Point Channels at any M3U through
the relay, and the channel list, numbers and guide all stay with the DVR:

| Endpoint | |
|---|---|
| `/api/m3u?url=<playlist>` | The playlist (cached 5 min) with every stream URL rewritten to `/api/stream`; `#EXTINF` attributes and names unchanged |
| `/api/stream?url=<stream>` | One source as HLS with an English WebVTT track (the same output as managed channels). Starts on first request, shared by all viewers, stops when idle |

Parameters (both endpoints; `/api/m3u` passes them on):

- `mt=seamless|nllb|whisper` backend (default Seamless). On `/api/m3u` a list,
  `mt=seamless,nllb,whisper`, lists each channel as N (first backend), N.1,
  N.2... named "... [NLLB]", same tvg-id. All backends of one source share one
  pull and one ASR session, so each extra backend only adds its own Mac time.
- `chno=9000&chno_step=0.001` puts the whole playlist under one number:
  9000.001, 9000.002... (decimals as in the step, 0.001 to 1; entries that
  would reach 9001 are left out). Backend variants (`mt=a,b`) simply take the
  next numbers. Overrides `chno_offset`.
- `chno_offset=K` adds K to every channel number (channels without a number
  are numbered in playlist order), e.g. one Channels source per backend.
- `lang=auto|he|ar|...` spoken language. With `auto` Whisper identifies it on
  the first speech chunks, then Seamless takes over (English is transcribed).
- `pos=bottom|middle|top` subtitle position.
- `compare=1` compare mode (backends burned in, with the relay's Compare
  settings).
- `chunk=long` long chunks: 20-28 s instead of 4-15 s, the video about 50 s
  behind. Whisper then handles about twice as many channels
  (docs/PERFORMANCE.md); for recordings rather than live viewing.
- Referrer and user agent from `#EXTVLCOPT` / `http-referrer=` in the playlist
  are passed on.

Example (Israel from iptv-org), as a Channels Custom Channels source, Stream
Format HLS:

```
http://relay-host:8095/api/m3u?url=https%3A%2F%2Fiptv-org.github.io%2Fiptv%2Fcountries%2Fil.m3u&lang=auto
```

Keep your own guide: leave the source's XMLTV setting on whatever you use
(the iptv-org playlists' tvg-ids match epgshare01 and iptv-org guides), or use
the relay's `/epg.xml`, which covers only managed channels.

Safety: sources must be http(s); `SOURCE_ALLOW=host.example,10.0.0.0/8`
restricts them to those hosts (a name also matches its subdomains) and
networks; at most `MAX_ADHOC_SESSIONS` (default 6) streams run at once (503
otherwise); a dead source gets 502. The relay has no authentication: keep it
on the LAN.

## Configuration (environment)

| Var | Default | |
|---|---|---|
| `PORT` | 8095 | |
| `PUBLIC_URL` | `http://localhost:PORT` | base URL written into the playlist |
| `ASR_BACKEND` | `remote` if `ASR_URL` is set, else `none` | relay: `remote` (an ASR server over HTTP, below) or `none` (video without subtitles). asr_server.py: `mlx` (Apple Silicon), `cuda` (NVIDIA, docs/NVIDIA.md) or `cpu` (no GPU, docs/CPU.md) |
| `ASR_URL` | | the ASR server, e.g. `http://asr-host:8096` |
| `ASR_TIMEOUT` | 15 | seconds per remote request |
| `DELAY` | 20 | target lead (s) of output behind the source when re-anchoring |
| `MIN_LEAD` | 15 | below this the output clock re-anchors to `DELAY` |
| `GRACE` | 60 | seconds a session keeps running after its last client leaves |
| `CHUNK_MIN`, `CHUNK_MAX`, `PAUSE_MS` | 4, 15, 300 | ASR chunks end at the first speech pause of `PAUSE_MS` after `CHUNK_MIN` seconds, or at the quietest point at `CHUNK_MAX` (i24: 7.5 s on average) |
| `HOLD_MIN_LEAD` | 15 | A sentence cut off at the end of a chunk waits for the next chunk while its cue is still this many seconds ahead of the output |
| `STALL` | 60 | restart a session whose input delivers nothing for this long |
| `MUSIC_CAPTION` | `[♪ music]` | caption for music without usable lyrics (empty = off) |
| `ARCHIVE_DIR`, `ARCHIVE_MINUTES` | off, 60 | rolling archive of ASR chunks for debugging (with "Keep audio archive" in the web page) |
| `MAX_ADHOC_SESSIONS` | 6 | stateless `/api/stream` streams at once |
| `EPG_TRANSLATE` | 0 | 1: translate `/epg.xml` into English with NLLB on the relay's CPU; the model (1.4 GB) downloads by itself into `models/` next to `CHANNELS_FILE` (`NLLB_MODEL` to put it elsewhere) |
| `LONG_DELAY` | 50 | video delay (s) of channels with long chunks |

## Speech coverage and the hallucination guard

The channels show colour bars with a tone outside broadcast hours. The goal is
captions for all speech, including loud and emotional delivery and songs, and
nothing over bars, tone or silence.

1. **Pre-gate** (relay, before ASR): skips chunks below -50 dBFS and chunks
   where more than 60% of the energy sits within ±10 Hz of a single frequency
   (a test tone).
2. **Decode** (backend): no VAD cut and no Whisper-internal no-speech/logprob
   skipping, because both silently dropped about 30% of KCTV speech. Each
   30 s window falls back from temperature 0 to 0.4 only when it loops
   (compression ratio > 2.4). A chunk that Silero VAD rates at least 50% speech
   but that decodes to nothing usable is retried once at 0.4/0.8. The backend
   reports the VAD `speech_ratio`.
3. **Segment filter** (relay, applied to any backend's output):
   - Always drops: no words, music notes, a blocklist of typical Whisper
     hallucinations ("Thank you for watching", "subscribe", ...), compression
     ratio > 2.4, and text repeated within the last 4 segments.
   - When `speech_ratio` < 0.3 (bars, music, noise), strictly drops
     `no_speech_prob > 0.6` together with `avg_logprob < -1`, and any
     `avg_logprob < -1.5`.
   - When there is speech, drops only `avg_logprob < -2`.
4. **Music**: a loud chunk with under 50% speech and no usable text gets a
   `[♪ music]` caption (`MUSIC_CAPTION`; set it empty to disable).

Measured on 5 minutes of live KCTV (news, impassioned speech, music): before
the change, 70% of speech time was captioned. After it, about 100% is captioned,
plus 52 s marked as music. Colour bars still produce no captions.

Tools (run inside the image, see the scripts' headers):

- `tools/guard_test.py <wav>`: chunks a recording like the relay does and shows
  what was kept or dropped.
- `tools/diag.py <wav>`: per chunk, shows the VAD ratio, the current output,
  the output with filters off, and the Korean transcript.
- `tools/coverage.py <wav>`: the share of speech time captioned, under the old
  and new policies.

With `ARCHIVE_DIR` set and "Keep audio archive" ticked in the web page (off by
default: it costs disk and CPU for every running channel), the relay keeps the last `ARCHIVE_MINUTES` of ASR chunks
(WAV, plus JSON with the raw segments, speech ratio and drop reasons) under
`ARCHIVE_DIR/<channel>/`. Use it to check a caption against the audio.

## Tests

```sh
docker build -t captions-dvr-tests -f tests/Dockerfile .
docker run --rm captions-dvr-tests                    # unit + source tests, about 8 min
docker run --rm captions-dvr-tests python -m pytest -q tests/unit   # a few seconds
```

`tests/unit` covers the relay's pieces (URL checks, `/api/stream` parsing,
start cards, cue timing, the guide translator's download, the HTTP
endpoints). `tests/sources` runs the relay against synthetic H.264, MPEG-2
and HEVC sources (live HLS and plain TS) and `tests/fake_asr.py`, a fake ASR
server with error modes: start card, segments, PAT/PMT, timed cues, the ASR
going down, a source dropping out.

## ASR HTTP contract

The relay's `remote` backend sends each audio chunk (6 to 12 s) as one request.
Any service that implements these two endpoints works. `relay/asr_server.py`
implements them with `ASR_BACKEND=mlx` (mlx-whisper on Apple Silicon; see
`mac/` and INSTALL.md), `ASR_BACKEND=cuda` on an NVIDIA GPU (tested on an RTX 3060;
docs/NVIDIA.md) or `ASR_BACKEND=cpu` without a GPU (one or two channels;
docs/CPU.md). The relay itself does no speech recognition.

### `POST /v1/translate?language=ko`

Request body: a WAV file, 16 kHz, mono, signed 16-bit little-endian PCM.
`Content-Type: audio/wav`.

Response `200 application/json`:

```json
{
  "segments": [
    {"start": 0.42, "end": 3.9, "text": " The fish moved freely.",
     "no_speech_prob": 0.03, "avg_logprob": -0.31, "compression_ratio": 1.22}
  ],
  "speech_ratio": 0.93,
  "duration": 11.98,
  "processing_time": 2.41
}
```

- The service translates the audio to English (Whisper `task=translate`, source
  language from `language`).
- `start` and `end` are seconds from the start of the posted audio. They must
  be accurate, because the relay uses them to time captions against the video.
- `text` is required. `no_speech_prob`, `avg_logprob` and `compression_ratio`
  are optional, but send them if the engine has them: the guard uses them.
- `speech_ratio` (optional): the fraction of the audio that VAD judges to be
  speech. Without it, the relay applies its strict filters to every chunk.
- Decode as described under "Speech coverage": don't cut by VAD and don't skip
  internally on no_speech/logprob. Use temperature 0, falling back only on
  compression ratio > 2.4, and don't condition on previous text.
- The service must answer well within `ASR_TIMEOUT` (15 s). A chunk that
  fails or times out is left uncaptioned.
- Several channels may send requests concurrently.
- `4xx` responses return `{"error": "..."}`.

### `POST /v1/multi?language=<code|auto>&tracks=nllb,whisper,original` (API 4)

All tracks for one chunk in one request. The server runs Whisper's
transcription, then NLLB on the CPU in parallel with Whisper's translation on
the GPU. Response: `{"tracks": {"nllb": [...], "whisper": [...], "original":
[...]}, "speech_ratio", "language", "errors", "processing_time"}`. The relay
uses it for the comparison tracks when `/health` reports `api >= 4`, after
the main (Seamless) request, so the main track never waits.

### `GET /health`

Returns `200 {"ok": true, "model": "..."}`.

## Running

### Docker (a Docker host)

The relay image (amd64 and arm64):

```sh
docker pull ghcr.io/hdhrgit/live-captions:latest
```

Or build it with compose: put `PUBLIC_URL` and `ASR_URL` in a `.env` next to
`compose.yaml`, then `docker compose up -d`. The guide translation's NLLB
model is read from the `captions-models` volume (`/models`).

### Native systemd service (Channels DVR box)

The remux relay with `ASR_BACKEND=remote` only needs Python 3.9 or newer
(with `venv`), PyAV and numpy. PyAV wheels bundle ffmpeg.

```sh
git clone <repository-url> && cd captions-dvr
sudo ./deploy/install.sh
sudo $EDITOR /etc/captions-dvr/captions-dvr.env   # PUBLIC_URL, ASR_URL
sudo $EDITOR /etc/captions-dvr/channels.json
sudo systemctl restart captions-dvr
journalctl -u captions-dvr -f
```

The installer does the following, and is safe to re-run to upgrade:

- creates the `captions-dvr` system user
- puts a venv in `/opt/captions-dvr/venv` (~210 MB) and the code in `/opt/captions-dvr/app`
- copies config to `/etc/captions-dvr/`, keeping any existing files
- installs and starts `captions-dvr.service`

## Capacity

How many channels one ASR server captions at once (an RTX 3060 and a Mac
mini M4), recommended settings and the benchmark tool (`bench/`):
docs/PERFORMANCE.md.

## Resource use (remux only, measured on a Docker host, i5-11400)

Measured with `ASR_BACKEND=remote` against a stub ASR server, on 1080p50 sources
at ~9 Mbps. The figures cover the HLS pull, AAC decode, resampling and WAV
upload, SEI injection on every frame, the TS mux, and one client.

| Channels | CPU (one i5-11400 core) | RAM |
|---|---|---|
| idle | 0% | 27 MB |
| 1 | 4% | 88 MB |
| 2 | 7% | 144 MB |
| 3 | 9% | 197 MB |

That is ~3 to 4% of a core and ~55 MB per channel. A Celeron J4125 is roughly
2.5 to 3x slower per core, so expect ~10 to 12% of one J4125 core per channel,
under half a core for all four.
