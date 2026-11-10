"""Start-card frame rates: NTSC rates are 1000/1001 of 24/30/60, PAL ones
stay 25/50 (24.975 fps once made the MPEG-2 encoder fail for a 25 fps source)."""
import io
from fractions import Fraction

import av
import pytest

import slate

BASE = {"width": 320, "height": 240, "sample_rate": 48000, "channels": 2, "profile": "Main"}


@pytest.mark.parametrize("fps,expect", [(24, Fraction(24000, 1001)), (25, 25), (30, Fraction(30000, 1001)),
                                        (50, 50), (60, Fraction(60000, 1001))])
def test_mpeg2_card_rates(monkeypatch, fps, expect):
    monkeypatch.setattr(slate, "SLATE_SECONDS", 2)
    segs = slate.render(dict(BASE, fps=fps, vcodec="mpeg2video", acodec="ac3"))
    v = av.open(io.BytesIO(segs[0][1])).streams.video[0]
    assert v.average_rate == expect


@pytest.mark.parametrize("fps", [25, 30, 50, 60])
def test_h264_card_rates(monkeypatch, fps):
    monkeypatch.setattr(slate, "SLATE_SECONDS", 2)
    segs = slate.render(dict(BASE, fps=fps))
    v = av.open(io.BytesIO(segs[0][1])).streams.video[0]
    assert v.average_rate == fps


def test_1080_mpeg2_card_is_interlaced(monkeypatch):
    monkeypatch.setattr(slate, "SLATE_SECONDS", 2)
    segs = slate.render(dict(BASE, width=1920, height=1080, fps=30, vcodec="mpeg2video", acodec="ac3"))
    frame = next(av.open(io.BytesIO(segs[0][1])).decode(video=0))
    assert frame.interlaced_frame


def test_card_key_changes_with_format():
    a = slate._key(dict(BASE, fps=25, vcodec="mpeg2video", acodec="ac3"))
    b = slate._key(dict(BASE, fps=30, vcodec="mpeg2video", acodec="ac3"))
    assert a != b and a.endswith("-ntsc")
