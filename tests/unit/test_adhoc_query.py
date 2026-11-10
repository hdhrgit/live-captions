"""/api/stream and /api/m3u query parsing (name= after url= once ended up in the source URL)."""
import urllib.parse

import adhoc

SRC = "https://cdn.example/live/master.m3u8"


def q(raw):
    return adhoc.parse_query(raw)


def test_encoded_url():
    assert q("url=" + urllib.parse.quote(SRC, safe=""))["url"] == [SRC]


def test_name_after_unencoded_url():
    r = q(f"url={SRC}&name=My+Channel&mt=whisper")
    assert r["url"] == [SRC]
    assert r["name"] == ["My Channel"] and r["mt"] == ["whisper"]


def test_unencoded_url_with_its_own_query():
    src = SRC + "?token=1&b=2"
    r = q(f"url={src}&lang=he")
    assert r["url"] == [src] and r["lang"] == ["he"]


def test_lost_equals_sign():
    assert q("url" + urllib.parse.quote(SRC, safe=""))["url"] == [SRC]


def test_chno_params_end_the_url():
    r = q(f"url={SRC}&chno=9000&chno_step=0.01")
    assert r["url"] == [SRC] and r["chno"] == ["9000"] and r["chno_step"] == ["0.01"]


def test_options_mt_lang_chunk():
    o, mt = adhoc.options(q(f"url={urllib.parse.quote(SRC, safe='')}&mt=sml&lang=he&chunk=long&name=X"))
    assert mt == "seamless"
    assert o["l"] == "he" and o["k"] == 1 and o["n"] == "X"
    c = adhoc.channel_cfg(adhoc.token(o))
    assert c["long_chunks"] and c["language"] == "he" and c["name"] == "X"


def test_options_rejects_bad_lang():
    import pytest
    with pytest.raises(adhoc.Refused):
        adhoc.options(q(f"url={urllib.parse.quote(SRC, safe='')}&lang=hebrew!"))
