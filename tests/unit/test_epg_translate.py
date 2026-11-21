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
    monkeypatch.setattr("asr.NLLB", lambda p, threads: Model())
    monkeypatch.setattr(epg, "CACHE", str(tmp_path / "cache.sqlite"))
    monkeypatch.setattr(epg, "NLLB_KEEP", 3600)
    assert t._ready() is True and t.mt is None   # loaded only for new strings
    assert t._load() is True and isinstance(t.mt, Model)


class Model:
    """A stand-in for asr.NLLB."""
    closed = False

    def translate(self, texts, lang):
        return [f"[{lang}] {t}" for t in texts]

    def close(self):
        self.closed = True


def test_model_loaded_for_new_strings_only_and_unloaded(monkeypatch, tmp_path):
    """Strings in the cache don't load the model; new ones do, and it is
    unloaded EPG_NLLB_KEEP after the job (freeing about 1.4 GB)."""
    import time
    import xml.etree.ElementTree as ET
    path = tmp_path / "m"
    path.mkdir()
    (path / "model.bin").write_bytes(b"x")
    monkeypatch.setenv("NLLB_MODEL", str(path))
    monkeypatch.setattr(epg, "CACHE", str(tmp_path / "cache.sqlite"))
    loads = []
    monkeypatch.setattr("asr.NLLB", lambda p, threads: loads.append(Model()) or loads[-1])
    monkeypatch.setattr(epg, "NLLB_KEEP", 0.5)
    t = epg.Translator()

    def progs():
        return [ET.fromstring('<programme channel="c"><title>新聞直播間</title></programme>')]
    assert t.translate_programmes(progs(), {"c": "zh"}) == 1 and len(loads) == 1
    for _ in range(40):
        if t.mt is None:
            break
        time.sleep(0.1)
    assert t.mt is None and loads[0].closed
    p = progs()
    assert t.translate_programmes(p, {"c": "zh"}) == 0 and len(loads) == 1   # from the cache
    assert p[0].find("title").text == "[zh] 新聞直播間"


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


class FakeTranslator:
    """Records what would be translated, per channel hint."""

    def __init__(self):
        self.seen = []

    def translate_programmes(self, elements, langs):
        for el in elements:
            for e in el.findall("title"):
                self.seen.append((el.get("channel"), langs.get(el.get("channel")), e.get("lang"), e.text))
            el.find("title").text = "EN " + el.find("title").text
        return len(elements)


def test_stateless_guide_languages():
    """/api/epg: lang attributes give each channel's language; "en"-labelled
    German is relabelled; Devanagari is recognised; English stays English."""
    xml = ('<tv><programme channel="a"><title lang="es">Noticias RCN</title></programme>'
           '<programme channel="b"><title lang="en">Der Tatort mit dem Kommissar und die Spur</title></programme>'
           '<programme channel="c"><title lang="en">आज तक की खबरें</title></programme>'
           '<programme channel="d"><title lang="en">The evening news with the anchor</title></programme></tv>')
    t = FakeTranslator()
    out, n = epg.translate_xmltv(xml.encode(), t)
    langs = {ch: (hint, lang) for ch, hint, lang, _ in t.seen}
    assert langs["a"] == ("es", "es")
    assert langs["b"] == ("de", "de")
    assert langs["c"][0] == "" and epg.text_language("आज तक की खबरें", "") == "hi"
    assert langs["d"] == ("", "en")
    assert n == 4 and b"EN Noticias RCN" in out and out.startswith(b"<?xml")


def test_guess_latin():
    assert epg.guess_latin("Les nouvelles du soir et la météo") == "fr"
    assert epg.guess_latin("Il telegiornale della sera con le notizie") == "it"
    assert epg.guess_latin("Tatort") is None


def test_stateless_guide_refetched_after_ttl(monkeypatch):
    """/api/epg: a copy older than EPG_ADHOC_MINUTES is fetched again (FAST
    guides cover only hours ahead); a fresh one is served from the cache."""
    import threading
    fetched = []
    g = epg.AdhocGuides(FakeTranslator())
    monkeypatch.setattr(g, "_fetch", lambda url: fetched.append(url) or
                        b'<tv><programme channel="a"><title lang="es">Hola amigos</title></programme></tv>')
    monkeypatch.setattr(threading, "Thread", lambda target, args, daemon: type(
        "T", (), {"start": lambda self: target(*args)})())
    g.get("http://x/epg.xml")
    assert b"EN Hola" in g.get("http://x/epg.xml") and len(fetched) == 1
    g.cache[("http://x/epg.xml", "")]["at"] -= g.TTL + 1
    g.get("http://x/epg.xml")
    assert len(fetched) == 2
