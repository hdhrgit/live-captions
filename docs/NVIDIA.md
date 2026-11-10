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

### Check it

In the same shell, before starting the server:

```sh
echo $LD_LIBRARY_PATH | tr ":" "\n" | grep nvidia | head -3
# .../site-packages/nvidia/cublas/lib
# .../site-packages/nvidia/cuda_cupti/lib
# .../site-packages/nvidia/cuda_nvrtc/lib

ls $SP/nvidia/cublas/lib $SP/nvidia/cudnn/lib | grep -E "libcublas.so|libcudnn.so"
# libcublas.so.12
# libcudnn.so.9

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce RTX 3060

python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
# 1
```

Then start the server and check it from the relay's machine:

```sh
curl http://<this-box>:8096/health
# {"ok": true, "api": 4, "mt": ["whisper", "nllb", "seamless"], "backend": "CudaWhisper", ...}
nvidia-smi    # the python processes hold several GB of GPU memory
```

Without the `LD_LIBRARY_PATH` line the server stops at start with
"CUDA libraries not found: libcublas.so.12, ..." and prints the two lines to
run. Older versions started anyway and failed on the first chunk with
"Unable to load any of {libcudnn_ops.so.9...}" or "Library libcublas.so.12
is not found or cannot be loaded". Fix it the same way.

The variable only lasts for that shell. To keep it, put the expanded value
in the service's environment: `Environment=LD_LIBRARY_PATH=...` in a systemd
unit (or its `EnvironmentFile`), or `environment:` in compose. Get the value
with `echo $LD_LIBRARY_PATH` after the export above.

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
      ASR_PARALLEL: "8"
      SEAMLESS_PARALLEL: "8"
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

(The CUDA image has the CUDA libraries on the loader path already.)
Check with `curl http://<this-box>:8096/health`: `"mt"` should list
`seamless` once the model loads, and `"parallel"` / `"batch"` show the
settings below.

## Many channels at once

A GPU decoding one chunk at a time stays half idle. The server decodes the
requests that arrive while a decode runs together, in one batch per model
(a lone request never waits):

| Setting | Default | Meaning |
|---|---|---|
| `ASR_PARALLEL` | 1 | Whisper/NLLB requests the server takes at once (8 on an NVIDIA GPU) |
| `SEAMLESS_PARALLEL` | `ASR_PARALLEL` | Seamless requests it takes at once (8) |
| `WHISPER_BATCH`, `SEAMLESS_BATCH` (or `ASR_BATCH` for both) | 8 | most requests in one batch |
| `WHISPER_WORKERS` | 2 | Whisper decodes side by side; more only made them compete for the GPU |

In the relay's web page, set "ASR parallel" to match (16 for an NVIDIA
server; 1 for a Mac). docs/PERFORMANCE.md has measured stream counts.

## Monitoring

Both the relay and the ASR server serve `GET /metrics` in the Prometheus
text format. For Netdata, add jobs to `go.d/prometheus.conf`:

```yaml
jobs:
  - name: captions_relay
    url: http://<relay-host>:8095/metrics
  - name: captions_asr
    url: http://<asr-host>:8096/metrics
```

Relay series are labelled by channel number (ad-hoc `/api/stream` streams
are summed); the ASR server reports requests, queue wait, decode time and
batch sizes per model, and GPU use where `nvidia-smi` is available.
