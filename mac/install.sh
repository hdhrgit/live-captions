#!/bin/sh
# Install or upgrade the captions-dvr ASR service on an Apple Silicon Mac.
# Run as the user that will own the service (not root), from a checkout:
#   ./mac/install.sh            # port 8096, whisper-large-v3
#   PORT=9000 MODEL=mlx-community/whisper-medium-mlx ./mac/install.sh
set -eu
SRC=$(cd "$(dirname "$0")/.." && pwd)
PREFIX="$HOME/captions-asr"
PORT=${PORT:-8096}
MODEL=${MODEL:-mlx-community/whisper-large-v3-mlx}
LABEL=com.captions-dvr.asr
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PY=${PYTHON:-python3}

[ "$(uname -m)" = arm64 ] || { echo "needs Apple Silicon (arm64)"; exit 1; }
mkdir -p "$PREFIX/app" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
[ -x "$PREFIX/venv/bin/python" ] || "$PY" -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install --quiet --upgrade pip
"$PREFIX/venv/bin/pip" install --quiet -r "$SRC/mac/requirements.txt"
cp "$SRC/relay/asr.py" "$SRC/relay/asr_server.py" "$PREFIX/app/"

# NLLB-200 1.3B for Korean/Arabic/Hebrew/... -> English text translation.
# Converted once to CTranslate2 int8 (~1.3 GB); needs torch/transformers
# only for the conversion.
NLLB_DIR="$PREFIX/nllb-200-distilled-1.3B-ct2"
if [ ! -f "$NLLB_DIR/model.bin" ]; then
    echo "Downloading and converting NLLB-200 (one-off, ~5.5 GB download)..."
    "$PREFIX/venv/bin/pip" install --quiet transformers torch
    # Fetch only the safetensors weights (not also pytorch_model.bin) into a
    # scratch dir, convert, then delete the download.
    TMP_NLLB="$PREFIX/nllb-download"
    "$PREFIX/venv/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/nllb-200-distilled-1.3B', local_dir='$TMP_NLLB',
                  allow_patterns=['*.json', '*.safetensors', 'sentencepiece.bpe.model'])"
    "$PREFIX/venv/bin/ct2-transformers-converter" --model "$TMP_NLLB" \
        --output_dir "$NLLB_DIR" --quantization int8 --force
    cp "$TMP_NLLB/sentencepiece.bpe.model" "$NLLB_DIR/"
    rm -rf "$TMP_NLLB" "$HOME/.cache/huggingface/hub/models--facebook--nllb-200-distilled-1.3B"
fi

# SeamlessM4T v2 large (speech -> English), the "English (Seamless)" track.
# ~11 GB, downloaded once (safetensors only). Skip with SEAMLESS=0.
SEAMLESS_DIR="$PREFIX/seamless-m4t-v2-large"
if [ "${SEAMLESS:-1}" = 1 ] && [ ! -f "$SEAMLESS_DIR/config.json" ]; then
    echo "Downloading SeamlessM4T v2 large (one-off, ~11 GB)..."
    "$PREFIX/venv/bin/pip" install --quiet transformers torch protobuf sentencepiece
    "$PREFIX/venv/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/seamless-m4t-v2-large', local_dir='$SEAMLESS_DIR.tmp',
                  allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])"
    mv "$SEAMLESS_DIR.tmp" "$SEAMLESS_DIR"
    rm -rf "$HOME/.cache/huggingface/hub/models--facebook--seamless-m4t-v2-large"
fi
if [ -f "$SEAMLESS_DIR/config.json" ]; then
    (cd "$PREFIX/app" && SEAMLESS_MODEL="$SEAMLESS_DIR" "$PREFIX/venv/bin/python" -c "
import time, numpy as np, asr
t = time.time(); m = asr.seamless(); print('Seamless loaded on', m.device, 'in %.0fs' % (time.time() - t))
t = time.time(); print('Seamless self-test:', repr(m.translate(np.zeros(16000 * 5, np.float32))), '%.1fs' % (time.time() - t))")
fi

echo "Downloading and warming up $MODEL (first run takes a while)..."
(cd "$PREFIX/app" && ASR_BACKEND=mlx MODEL="$MODEL" NLLB_MODEL="$NLLB_DIR" "$PREFIX/venv/bin/python" -c "
import numpy as np, asr
print(asr.from_env().transcribe(np.zeros(16000 * 5, np.float32)))
print('NLLB:', asr.nllb().translate(['안녕하세요'], 'ko'))")

sed -e "s|@PREFIX@|$PREFIX|g" -e "s|@PORT@|$PORT|g" -e "s|@MODEL@|$MODEL|g" -e "s|@NLLB@|$NLLB_DIR|g" -e "s|@SEAMLESS@|$SEAMLESS_DIR|g" \
    -e "s|@LOG@|$HOME/Library/Logs/captions-asr.log|g" "$SRC/mac/$LABEL.plist" > "$PLIST"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
sleep 3
curl -fsS "http://localhost:$PORT/health" && echo && echo "OK: listening on port $PORT"
