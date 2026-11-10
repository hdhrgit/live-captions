"""The relay's HTTP endpoints, against a real relay process with no ASR:
an empty start, playlist.m3u = HLS with a subtitle rendition, no .ts routes."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

RELAY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "relay")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def relay(tmp_path):
    def start(channels=None, **env):
        port = free_port()
        cfile = tmp_path / "channels.json"
        if channels is not None:
            cfile.write_text(json.dumps(channels))
        e = dict(os.environ, PORT=str(port), CHANNELS_FILE=str(cfile), ASR_BACKEND="none",
                 PUBLIC_URL=f"http://127.0.0.1:{port}")
        e.pop("ASR_URL", None)
        e.update(env)
        proc = subprocess.Popen([sys.executable, "relay.py"], cwd=RELAY, env=e,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append(proc)
        base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                urllib.request.urlopen(base + "/status", timeout=1)
                return base, cfile, proc
            except OSError:
                time.sleep(0.2)
        raise RuntimeError("relay didn't start: " + proc.stdout.read(4000).decode())
    procs = []
    yield start
    for p in procs:
        p.terminate()
        p.wait(10)


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


CHANNELS = {"sources": [], "settings": {}, "channels": [
    {"id": "t1", "name": "Test One", "number": 9001, "url": "http://127.0.0.1:1/none.m3u8", "source": "manual",
     "enabled": True, "language": "en", "tvg": {}, "headers": {}}]}


def test_empty_start(relay):
    base, cfile, _ = relay()
    assert json.loads(get(base + "/status")[1])["channels"] == {}
    code, body = get(base + "/playlist.m3u")
    assert code == 200 and body.strip().startswith("#EXTM3U") and "#EXTINF" not in body
    assert "<tv" in get(base + "/epg.xml")[1]
    assert "No channels yet" in get(base + "/")[1]
    assert json.loads(cfile.read_text())["channels"] == []


def test_playlists_are_hls(relay):
    base, _, _ = relay(CHANNELS)
    for name in ("playlist.m3u", "playlist-hls.m3u"):
        code, body = get(base + "/" + name)
        assert code == 200 and f"{base}/hls/t1/master.m3u8" in body and 'tvg-chno="9001"' in body


def test_ts_routes_are_gone(relay):
    base, _, _ = relay(CHANNELS)
    for path in ("/t1.ts", "/stream.ts", "/playlist-ts.m3u"):
        assert get(base + path)[0] == 404


def test_master_has_subtitles(relay):
    base, _, _ = relay(CHANNELS)
    code, body = get(base + "/hls/t1/master.m3u8")
    assert code == 200 and "TYPE=SUBTITLES" in body and "subs.m3u8" in body


def test_status_has_version_and_problems(relay):
    base, _, _ = relay()
    st = json.loads(get(base + "/status")[1])
    assert "version" in st and st["config_problems"] == [] and st["epg_model"] == {"state": "off"}


def test_bad_asr_url_is_reported(relay):
    base, _, proc = relay(ASR_BACKEND="remote", ASR_URL="http:/asr-host.lan:8096")
    st = json.loads(get(base + "/status")[1])
    assert any("ASR_URL" in p for p in st["config_problems"])
    proc.terminate()
    out = proc.stdout.read().decode()
    assert "configuration: ASR_URL='http:/asr-host.lan:8096'" in out
