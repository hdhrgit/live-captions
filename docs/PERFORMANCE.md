# Performance

How many channels one ASR server can caption at once, measured with
`bench/ramp.py` (live public streams, tuned through the relay's
`/api/stream`, about two minutes measured per step). "Streams" counts every
stream with a running session; "skipped" are chunks that went uncaptioned
because the ASR answer would have come too late. A step is fine while
nothing is skipped and the ASR time stays well under the relay's lead
(about 20 s).

The relay itself is not the limit: each channel runs in its own process,
about 3-6 % of a CPU core per stream (PyAV demuxing, audio decoding,
segmenting).

## NVIDIA RTX 3060 12 GB

ASR server with `ASR_BACKEND=cuda`, `ASR_PARALLEL=8`, `SEAMLESS_PARALLEL=8`,
batching on (`WHISPER_BATCH`/`SEAMLESS_BATCH` 8); relay "ASR parallel" 16.

### Whisper large-v3, English (transcription)

| Streams | ASR median / p90 | Skipped / made | GPU busy |
|---|---|---|---|
| 11 | 1.3 s / 3.1 s | 0 / 127 | 67 % |
| 17 | 5.3 s / 7.6 s | 13 / 235 | 77 % |
| 22 | 8.7 s / 11.5 s | 80 / 91 | 77 % |
| 25 | 11.2 s / 13.6 s | 160 / 64 | 66 % |

About **14-15 streams**. Whisper's encoder always processes a 30 s window,
so a 6 s chunk costs as much as a 30 s one; batching barely helps, and
more than 2 parallel decoders (`WHISPER_WORKERS`) only made them compete.
Long chunks fix that (next table).

### Seamless M4T v2 (translation to English)

| Streams | ASR median / p90 | Skipped / made | GPU busy |
|---|---|---|---|
| 11 | 2.1 s / 3.7 s | 2 / 119 | 66 % |
| 14 | 5.1 s / 6.6 s | 0 / 186 | 53 % |
| 17 | 6.3 s / 9.4 s | 59 / 107 | 50 % |

About **14 streams** with batching; one request at a time it was about 8
(at 11 streams half the chunks were skipped).

GPU memory: about 8-10 GB with Whisper large-v3 and Seamless loaded.

### Whisper large-v3 with long chunks

| Streams | ASR median / p90 | Skipped / made | GPU busy |
|---|---|---|---|
| 12 | 1.6 s / 4.0 s | 1 / 233 | 55 % |
| 18 | 3.0 s / 4.8 s | 1 / 429 | 67 % |
| 25 | 3.5 s / 5.7 s | 0 / 540 | 82 % |
| 29 | 6.9 s / 9.6 s | 6 / 699 | 80 % |

About **30 streams**, twice as many, for about 30 s more delay (below).

## Mac mini M4

`mac/install.sh` (Whisper large-v3 on MLX, Seamless and NLLB on the GPU via
PyTorch MPS), relay "ASR parallel" 1: the Mac's GPU decodes one chunk at a
time, whichever model, so its streams share one queue.

### Seamless M4T v2 only

| Seamless streams | ASR median / p90 | Skipped / made |
|---|---|---|
| 1-5 | | 1 / 179 |
| 6 | 5.1 s / 15.5 s | 0 / 120 |
| 7 | 9.4 s / 15.9 s | 12 / 92 |
| 8 | 9.0 s / 18.5 s | 31 / 84 |

About **6 Seamless channels**.

### Whisper large-v3, English

| Streams | ASR median / p90 | Skipped / made |
|---|---|---|
| 5 | 6.1 s / 14.0 s | 12 / 83 |
| 7 | 7.9 s / 12.0 s | 60 / 40 |
| 9 | 7.9 s / 11.2 s | 95 / 22 |

About **4 Whisper channels** (Whisper large-v3 on MLX costs the Mac more
than Seamless). Mixed, the two share the queue: count about 1.5 Seamless
channels per Whisper channel. Long chunks don't help on the Mac (at 7
streams: 15 s median, 18 of 88 chunks skipped).

## CPU only (Intel i5-11400, 8 of its 12 threads)

`ASR_BACKEND=cpu` (docs/CPU.md): Whisper large-v3-turbo in int8; foreign
languages are transcribed and translated by NLLB, also on the CPU. No
Seamless.

| Streams | English | English, long chunks | Foreign (turbo + NLLB) |
|---|---|---|---|
| 1 | 4.5 s, 0 skipped | 4.7 s, 0 skipped | 8.6 s, 0 skipped |
| 2 | 4.4 s, 0 skipped | 4.7 s, 0 skipped | 12.7 s, 19 of 26 skipped |
| 3 | 8.7 s, 9 of 46 skipped | | |
| 4 | 9.1 s, 10 of 52 skipped | 6.6 s, 0 skipped | |
| 6 | | 11.0 s, 2 of 87 skipped | |

(ASR median per chunk.) About **2 English channels, 5 with long chunks, or
1 foreign channel**: enough for one or two channels without a GPU.

## Recommended settings

| | Mac (Apple Silicon) | NVIDIA |
|---|---|---|
| relay "ASR parallel" (web page) | 1 | 16 |
| `ASR_PARALLEL` (ASR server) | 1 | 8 |
| `SEAMLESS_PARALLEL` | 1 | 8 |
| `WHISPER_BATCH`, `SEAMLESS_BATCH` | 8 (unused at 1) | 8 |
| `WHISPER_WORKERS` | - | 2 |

## Long chunks

The channel option `long_chunks` (stateless: `chunk=long`) cuts 20-28 s
chunks instead of 4-15 s and runs the video about 50 s behind the source
(the start-up card covers what the source's backlog doesn't). Whisper then
does about a quarter of the encoder passes for the same audio; captions are
still sentence-sized, from Whisper's own segment timestamps. Meant for
recordings and anything else where the extra delay doesn't matter; Seamless
gains little (its cost follows the audio length).

## Running the benchmark

```sh
bench/ramp.py --relay http://<relay-host>:8095 --asr http://<asr-host>:8096 \
    --group english --mt whisper --lang en --steps 4,8,12,16
bench/ramp.py --relay http://<relay-host>:8095 --asr http://<asr-host>:8096 \
    --group foreign --mt seamless --steps 4,8,12,16
```

`bench/streams.json` lists public test streams; ones that don't answer are
skipped, and each run picks a random subset (`--seed` to repeat one).
