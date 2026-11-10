#!/bin/sh
# Install or upgrade the captions-dvr relay as a native systemd service.
# Run as root from a checkout:  sudo ./deploy/install.sh
# Needs python3 (>= 3.9) with venv; PyAV wheels bundle ffmpeg, so no
# system ffmpeg is required. ASR is remote, so faster-whisper is not installed.
set -eu
SRC=$(cd "$(dirname "$0")/.." && pwd)
PREFIX=/opt/captions-dvr
CONF=/etc/captions-dvr

id captions-dvr >/dev/null 2>&1 || useradd --system --home-dir "$PREFIX" --shell /usr/sbin/nologin captions-dvr
mkdir -p "$PREFIX/app" "$CONF"
# Burned-in subtitles: a font, and FriBiDi for right-to-left text.
command -v apt-get >/dev/null && apt-get install -y --no-install-recommends libfribidi0 fonts-dejavu-core
[ -x "$PREFIX/venv/bin/python" ] || python3 -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install --upgrade pip
"$PREFIX/venv/bin/pip" install -r "$SRC/relay/requirements.txt"
cp "$SRC"/relay/*.py "$SRC"/relay/*.json "$SRC/relay/ui.html" "$PREFIX/app/"
# channels.json lives in /var/lib/captions-dvr (written by the web page);
# a new install starts with no channels.
[ -f "$CONF/captions-dvr.env" ] || cp "$SRC/deploy/captions-dvr.env" "$CONF/"
"$PREFIX/venv/bin/python" -c "import sys; sys.path.insert(0, '$PREFIX/app'); import av, numpy, cc608, asr; print('ok: PyAV', av.__version__)"

if [ -d /run/systemd/system ]; then
    cp "$SRC/deploy/captions-dvr.service" /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable captions-dvr
    systemctl restart captions-dvr
    systemctl --no-pager status captions-dvr | head -5
else
    echo "systemd not running; copy deploy/captions-dvr.service to /etc/systemd/system yourself"
fi
echo "Edit $CONF/captions-dvr.env (PUBLIC_URL, ASR_URL), then: systemctl restart captions-dvr"
