"""EPG_TRANSLATE: off = no model, no download, no log; on = a missing model
downloads (retries, then ready), and the translator picks it up."""
import logging

import epg
import nllbget


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("EPG_TRANSLATE", raising=False)
    assert epg.TRANSLATE() is False
    monkeypatch.setenv("EPG_TRANSLATE", "1")
    assert epg.TRANSLATE() is True


def test_missing_model_starts_download(monkeypatch, tmp_path):
    monkeypatch.setenv("NLLB_MODEL", str(tmp_path / "none"))
    called = []
    monkeypatch.setattr(epg, "MODEL_MISSING", lambda: called.append(1))
    t = epg.Translator()
    assert t._ready() is False and called == [1]


def test_model_appearing_later_is_used(monkeypatch, tmp_path):
    path = tmp_path / "m"
    monkeypatch.setenv("NLLB_MODEL", str(path))
    monkeypatch.setattr(epg, "MODEL_MISSING", lambda: None)
    t = epg.Translator()
    assert t._ready() is False
    path.mkdir()
    (path / "model.bin").write_bytes(b"x")
    monkeypatch.setattr("asr.NLLB", lambda p, threads: "loaded")
    monkeypatch.setattr(epg, "CACHE", str(tmp_path / "cache.sqlite"))
    assert t._ready() is True and t.mt == "loaded"


def test_download_retries_then_ready(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("NLLB_MODEL", str(tmp_path / "m"))
    attempts = []

    def fake_fetch(dest, progress, phase):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("network down")
        progress(nllbget.TOTAL, "model.bin")

    monkeypatch.setattr(nllbget, "fetch", fake_fetch)
    monkeypatch.setattr(nllbget.Download, "RETRIES", (0, 0, 0))
    d, done = nllbget.Download(), []
    with caplog.at_level(logging.INFO, logger="epg"):
        d.run(lambda: done.append(1))
    assert d.state["state"] == "ready" and done == [1] and len(attempts) == 3
    assert sum("retrying" in r.message for r in caplog.records) == 2


def test_download_gives_up_after_retries(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("NLLB_MODEL", str(tmp_path / "m"))
    monkeypatch.setattr(nllbget, "fetch", lambda *a: (_ for _ in ()).throw(OSError("dead")))
    monkeypatch.setattr(nllbget.Download, "RETRIES", (0, 0))
    d = nllbget.Download()
    with caplog.at_level(logging.WARNING, logger="epg"):
        d.run(None)
    assert d.state["state"] == "failed" and "dead" in d.state["error"]
    assert any("download failed (OSError: dead); will retry at the next guide refresh, or run "
               "captions-dvr-get-nllb now" in r.message for r in caplog.records)


def test_fetch_resumes_and_checks(monkeypatch, tmp_path):
    """A .part file resumes with a Range request; a bad checksum is deleted."""
    import hashlib
    data = b"abcdefghij" * 100
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(nllbget, "FILES", [("http://x/f", "f.bin", len(data), sha)])
    (tmp_path / "f.bin.part").write_bytes(data[:400])
    seen = {}

    class Resp:
        status = 206

        def __init__(self, req):
            seen["range"] = req.headers.get("Range")
            self.left = data[400:]

        def read(self, n):
            out, self.left = self.left[:n], self.left[n:]
            return out

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(nllbget.urllib.request, "urlopen", lambda req, timeout: Resp(req))
    nllbget.fetch(str(tmp_path))
    assert seen["range"] == "bytes=400-" and (tmp_path / "f.bin").read_bytes() == data


def test_chinese_labelled_english_is_translated(tmp_path):
    """A guide's lang="en" on Chinese text (CCTV-13's) doesn't stop the
    translation; English and Latin text labelled "en" stay as they are."""
    import sqlite3
    import xml.etree.ElementTree as ET

    class MT:
        def translate(self, texts, lang):
            return [f"[{lang}] {t}" for t in texts]

    tr = epg.Translator()
    tr.mt, tr.codes = MT(), {"zh": "zho_Hans"}
    tr.db = sqlite3.connect(str(tmp_path / "tr.db"))
    tr.db.execute("CREATE TABLE tr2 (lang TEXT, src TEXT, en TEXT, PRIMARY KEY (lang, src))")
    tr._ready = lambda: True
    progs = [ET.fromstring('<programme channel="c13"><title lang="en">新聞直播間</title></programme>'),
             ET.fromstring('<programme channel="c1"><title lang="en">Network News</title></programme>')]
    assert tr.translate_programmes(progs, {"c13": "zh", "c1": "zh"}) == 1
    assert progs[0].find("title").text == "[zh] 新聞直播間"
    assert progs[0].find("desc").text == "Original title: 新聞直播間"
    assert progs[1].find("title").text == "Network News"
