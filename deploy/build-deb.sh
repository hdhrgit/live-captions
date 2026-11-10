#!/bin/sh
# Build the relay as a Debian package:  deploy/build-deb.sh [version]  -> dist/captions-dvr_<version>_all.deb
# Architecture "all": the Python packages (PyAV with its own ffmpeg, numpy,
# Pillow, CTranslate2) are installed into a venv by postinst, as wheels for
# the machine's own architecture, so installing needs internet access.
set -eu
umask 022
SRC=$(cd "$(dirname "$0")/.." && pwd)
# Version: the argument, else $VERSION, else from git: 0.1.0 on a tag,
# 0.1.0+g6b33b4a after it (Debian sorts that above 0.1.0).
VERSION=${1:-${VERSION:-$("$SRC/deploy/version.sh")}}
PKG=$(mktemp -d) && chmod 755 "$PKG"
trap 'rm -rf "$PKG"' EXIT

mkdir -p "$PKG/DEBIAN" "$PKG/opt/captions-dvr/app" "$PKG/lib/systemd/system" "$PKG/etc/captions-dvr"
cp "$SRC"/relay/*.py "$SRC"/relay/*.json "$SRC/relay/ui.html" "$SRC/relay/requirements.txt" "$PKG/opt/captions-dvr/app/"
echo "$VERSION" > "$PKG/opt/captions-dvr/app/VERSION"   # shown in /status and the web page
cp "$SRC/deploy/captions-dvr.service" "$PKG/lib/systemd/system/"
cp "$SRC/deploy/captions-dvr.env" "$PKG/etc/captions-dvr/"
mkdir -p "$PKG/usr/bin"
cat > "$PKG/usr/bin/captions-dvr-get-nllb" <<'SH'
#!/bin/sh
# Download the guide translator (NLLB-200, 1.4 GB) for the relay; resumes.
set -e
set -a; . /etc/captions-dvr/captions-dvr.env; set +a
cd /opt/captions-dvr/app
if [ "$(id -u)" = 0 ]; then
    exec runuser -u captions-dvr -- /opt/captions-dvr/venv/bin/python nllbget.py "$@"
fi
exec /opt/captions-dvr/venv/bin/python nllbget.py "$@"
SH
chmod 755 "$PKG/usr/bin/captions-dvr-get-nllb"

cat > "$PKG/DEBIAN/control" <<EOF
Package: captions-dvr
Version: $VERSION
Architecture: all
Maintainer: hdhrgit <hdhrgit@users.noreply.github.com>
Depends: python3 (>= 3.9), python3-venv, libfribidi0, fonts-dejavu-core, adduser
Section: video
Priority: optional
Description: English subtitles for live TV in Channels DVR
 Relays HLS and MPEG-TS channels with English WebVTT subtitles, from speech
 recognition and translation on a separate ASR server (Mac or NVIDIA GPU).
EOF
echo /etc/captions-dvr/captions-dvr.env > "$PKG/DEBIAN/conffiles"

cat > "$PKG/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
[ "$1" = configure ] || exit 0
id captions-dvr >/dev/null 2>&1 || adduser --system --group --home /var/lib/captions-dvr --no-create-home captions-dvr
mkdir -p /var/lib/captions-dvr && chown captions-dvr:captions-dvr /var/lib/captions-dvr
echo "captions-dvr: installing Python packages into /opt/captions-dvr/venv (needs internet, takes a few minutes)..."
[ -x /opt/captions-dvr/venv/bin/python ] || python3 -m venv /opt/captions-dvr/venv
/opt/captions-dvr/venv/bin/pip install --disable-pip-version-check --progress-bar off -r /opt/captions-dvr/app/requirements.txt
/opt/captions-dvr/venv/bin/python -c "import av, numpy, PIL; print('captions-dvr: Python packages done (PyAV', av.__version__ + ')')"
if [ -d /run/systemd/system ]; then
    systemctl daemon-reload
    systemctl enable captions-dvr
    # Start (or restart after an upgrade) only once the ASR server is set.
    if grep -Eq '^ASR_URL=http' /etc/captions-dvr/captions-dvr.env && ! grep -q '^ASR_URL=http://asr-host' /etc/captions-dvr/captions-dvr.env; then
        systemctl restart captions-dvr
    else
        echo "captions-dvr: set PUBLIC_URL and ASR_URL in /etc/captions-dvr/captions-dvr.env, then: systemctl restart captions-dvr"
    fi
fi
EOF

cat > "$PKG/DEBIAN/prerm" <<'EOF'
#!/bin/sh
set -e
if [ -d /run/systemd/system ] && [ "$1" = remove ]; then
    systemctl stop captions-dvr || true
    systemctl disable captions-dvr || true
fi
EOF

cat > "$PKG/DEBIAN/postrm" <<'EOF'
#!/bin/sh
set -e
case "$1" in
    remove)
        rm -rf /opt/captions-dvr/venv
        find /opt/captions-dvr/app -name __pycache__ -type d -exec rm -rf {} + || true
        [ -d /run/systemd/system ] && systemctl daemon-reload || true ;;
    purge)
        rm -rf /opt/captions-dvr /var/lib/captions-dvr /etc/captions-dvr
        id captions-dvr >/dev/null 2>&1 && deluser --system captions-dvr || true ;;
esac
EOF
chmod 755 "$PKG/DEBIAN/postinst" "$PKG/DEBIAN/prerm" "$PKG/DEBIAN/postrm"
mkdir -p "$SRC/dist"
dpkg-deb --root-owner-group --build "$PKG" "$SRC/dist/captions-dvr_${VERSION}_all.deb" >&2   # stdout: the path only
echo "$SRC/dist/captions-dvr_${VERSION}_all.deb"
