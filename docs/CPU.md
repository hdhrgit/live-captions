# ASR server without a GPU

`relay/asr_server.py` with `ASR_BACKEND=cpu` runs Whisper (faster-whisper,
int8) and NLLB on the CPU. Seamless is left out: it is far too slow on a
CPU for live captions, so foreign languages are transcribed by Whisper and
translated by NLLB. Expect one or two channels (docs/PERFORMANCE.md).

```sh
python3 -m venv ~/captions-asr && . ~/captions-asr/bin/activate
pip install -r relay/requirements-cpu.txt
cd relay
ASR_BACKEND=cpu MODEL=large-v3-turbo ASR_THREADS=8 MODEL_DIR=~/models \
  NLLB_MODEL=~/models/nllb-200-distilled-1.3B-ct2 PORT=8096 python asr_server.py
```

| Setting | Default | |
|---|---|---|
| `MODEL` | `large-v3-turbo` | a faster-whisper model; downloaded to `MODEL_DIR` on first start. `large-v3-turbo` transcribes well but doesn't translate, which NLLB does |
| `ASR_THREADS` | all cores | CPU threads for Whisper; leave a few for the relay if it runs on the same box |
| `NLLB_MODEL`, `NLLB_THREADS` | | a CTranslate2 conversion of `facebook/nllb-200-distilled-1.3B` (see `mac/install.sh` for the conversion), its threads (4) |

Then set the relay's `ASR_URL=http://<this-box>:8096` and leave "ASR
parallel" at 1. For more channels per CPU, use long chunks (`long_chunks`
per channel, `chunk=long` on `/api/stream`), which cost about a quarter of
the Whisper work for the same audio.
