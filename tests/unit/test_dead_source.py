"""A channel whose stream moved or died: a source refresh follows the moved
URL (by tvg-id), a URL set by hand stays, and a cold tune of a feed that
doesn't answer fails at once (so Channels DVR tunes the alternate channel)."""
import http.server
import threading
import time

import pytest

import config
import relay


def entry(url, tid="x.us@SD", name="X News"):
    return {"url": url, "name": name, "tvg": {"tvg-id": tid}, "headers": {}}


def cfg_with(tmp_path, channels):
    c = config.Config(str(tmp_path / "channels.json"))
    c.sources = [{"id": "s", "name": "s", "url": "http://list"}]
    c.channels = channels
    return c


def chan(url, **kw):
    return dict({"id": "x", "source": "s", "name": "X News", "url": url, "number": 8525, "enabled": True,
                 "headers": {}, "tvg": {"tvg-id": "x.us@SD"}, "health": {"ok": False}}, **kw)


def test_enabled_channel_follows_moved_url(tmp_path):
    c = cfg_with(tmp_path, [chan("http://old/x.m3u8")])
    c.follow_moved("s", [entry("http://new/x.m3u8"), entry("http://other/y.m3u8", "y.us@SD")])
    ch = c.channels[0]
    assert ch["url"] == "http://new/x.m3u8" and ch["number"] == 8525 and "health" not in ch


def test_name_decides_between_feeds(tmp_path):
    c = cfg_with(tmp_path, [chan("http://old/x.m3u8")])
    c.follow_moved("s", [entry("http://a/x.m3u8", name="X News (720p)"), entry("http://b/x.m3u8")])
    assert c.channels[0]["url"] == "http://b/x.m3u8"


@pytest.mark.parametrize("kw", [{"url_manual": True}, {"enabled": False}, {"source": "other"}])
def test_hand_set_disabled_or_other_source_stays(tmp_path, kw):
    c = cfg_with(tmp_path, [chan("http://old/x.m3u8", **kw)])
    c.follow_moved("s", [entry("http://new/x.m3u8")])
    assert c.channels[0]["url"] == "http://old/x.m3u8"


def test_url_set_by_hand(tmp_path):
    c = cfg_with(tmp_path, [chan("http://old/x.m3u8")])
    ch = c.update_channel("x", {"url": " http://fast/play/x.m3u8 "})
    assert ch["url"] == "http://fast/play/x.m3u8" and ch["url_manual"] and ch["number"] == 8525
    with pytest.raises(ValueError):
        c.update_channel("x", {"url": "not a url"})


@pytest.fixture
def upstream():
    """An HLS feed: /ok.m3u8 answers, anything else is 404."""
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/ok.m3u8" else 404)
            self.end_headers()
            self.wfile.write(b"#EXTM3U\n")

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


class Ch:
    def __init__(self, urls):
        self.cfg, self._urls = {"headers": {}}, urls

    def urls(self):
        return self._urls


def test_source_reachable(upstream):
    assert relay.source_reachable(Ch([f"{upstream}/ok.m3u8"]))
    assert not relay.source_reachable(Ch([f"{upstream}/gone.m3u8"]))
    assert relay.source_reachable(Ch([f"{upstream}/gone.m3u8", f"{upstream}/ok.m3u8"]))   # failover feed
    assert not relay.source_reachable(Ch(["http://127.0.0.1:9/x.m3u8"]))
    assert relay.source_reachable(Ch(["http://tuner:5004/auto/v2.1"]))   # a tuner isn't opened to check


def test_source_dead_after_timeout(monkeypatch):
    class S:
        stats, last_input = {"input": "error: 404"}, time.monotonic()
    ch = type("C", (), {"session": S()})()
    assert not relay.source_dead(ch)
    monkeypatch.setattr(relay, "DEAD_SOURCE_TIMEOUT", 0)
    assert relay.source_dead(ch)
    S.stats = {"input": "streaming"}
    assert not relay.source_dead(ch)
