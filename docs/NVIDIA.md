# ASR server on an NVIDIA GPU

Tested on an RTX 3060 12 GB. Reports and fixes from anyone with other NVIDIA
cards are welcome.

The relay does no speech recognition itself; it sends audio to an ASR server
over HTTP (README, "ASR HTTP contract"). On a Mac that's `mac/install.sh`.
On a Linux box with an NVIDIA GPU, `relay/asr_server.py` can run with
`ASR_BACKEND=cuda`:

- Whisper: faster-whisper on the GPU (`MODEL`, default `large-v3`, float16).
- Seamless (the default translation): SeamlessM4T v2 via transformers, on
  the GPU in float16 when CUDA is available. Point `SEAMLESS_MODEL` at a
  download of `facebook/seamless-m4t-v2-large`.
- NLLB (optional): `NLLB_MODEL` at a CTranslate2 conversion of
  `facebook/nllb-200-distilled-1.3B`, `NLLB_DEVICE=cuda`.

Roughly 10-12 GB of GPU memory for Whisper large-v3 and Seamless together.

## Running it directly

```sh
python3 -m venv ~/captions-asr && . ~/captions-asr/bin/activate
pip install -r relay/requirements-cuda.txt   # a CUDA build of torch may need the
                                             # index URL from pytorch.org
cd relay
# faster-whisper (CTranslate2) needs the CUDA libraries that come with torch:
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr "\n" ":")$LD_LIBRARY_PATH
ASR_BACKEND=cuda SEAMLESS_MODEL=/models/seamless-m4t-v2-large \
  NLLB_MODEL=/models/nllb-200-distilled-1.3B-ct2 NLLB_DEVICE=cuda \
  PORT=8096 python asr_server.py
```

Then set the relay's `ASR_URL=http://<this-box>:8096`.

## With Docker (NVIDIA Container Toolkit)

```yaml
services:
  asr:
    image: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
    working_dir: /app
    command: >
      bash -c "apt-get update && apt-get install -y python3-pip &&
               pip3 install -r requirements-cuda.txt && python3 asr_server.py"
    environment:
      ASR_BACKEND: cuda
      SEAMLESS_MODEL: /models/seamless-m4t-v2-large
      NLLB_MODEL: /models/nllb-200-distilled-1.3B-ct2
      NLLB_DEVICE: cuda
      MODEL_DIR: /models
    volumes:
      - ./relay:/app:ro
      - ./models:/models
    ports:
      - "8096:8096"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Check with `curl http://<this-box>:8096/health`: `"mt"` should list
`seamless` once the model loads.
