"""ASR_URL / PUBLIC_URL validation (an install had ASR_URL=http:/asr-host.lan:8096)."""
import pytest

import relay


@pytest.mark.parametrize("value", ["http:/asr-host.lan:8096", "http//relay:8095", "asr-host.lan:8096", "", "ftp://x:1"])
def test_bad_urls_are_named(value):
    problem = relay.url_problem("ASR_URL", value)
    assert problem and repr(value) in problem


@pytest.mark.parametrize("value", ["http://asr-host.lan:8096", "https://asr.example:443", "http://10.0.0.5:8096/"])
def test_good_urls(value):
    assert relay.url_problem("ASR_URL", value, need_port=True) is None


def test_asr_url_needs_a_port():
    assert "no port" in relay.url_problem("ASR_URL", "http://asr-host.lan", need_port=True)


def test_public_url_without_port_is_fine():
    assert relay.url_problem("PUBLIC_URL", "http://dvr.example") is None
