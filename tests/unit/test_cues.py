"""Cue timing and layout: a cue never stays up past its reading time
(15 chars/s + 1 s, 1.5-7 s), sentences are split, lines are broken."""
import cc608

T = 90000


def test_long_gap_is_capped_at_max_show():
    (a, b, _), = cc608.timed_groups(0, 60 * T, [["Short line."]])
    assert (b - a) / T <= cc608.MAX_SHOW


def test_short_text_stays_min_show():
    (a, b, _), = cc608.timed_groups(0, int(0.2 * T), [["Hi."]])
    assert (b - a) / T >= cc608.MIN_SHOW


def test_reading_time_cap():
    text = "x" * 30
    (a, b, _), = cc608.timed_groups(0, 60 * T, [[text]])
    assert abs((b - a) / T - (30 / cc608.LINGER_CPS + 1.0)) < 0.01


def test_groups_follow_each_other():
    groups = [["First sentence here."], ["Second one."]]
    out = cc608.timed_groups(0, 6 * T, groups)
    assert out[0][0] == 0 and out[1][0] > out[0][0]


def test_sentences_split():
    assert cc608.sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]


def test_layout_lines_fit_width():
    groups = cc608.layout_text("This is a fairly long sentence that has to be broken into lines for the screen.",
                               32, 2)
    assert all(len(line) <= 32 for g in groups for line in g)
    assert all(len(g) <= 2 for g in groups)


def test_ends_sentence():
    assert cc608.ends_sentence("Done.") and not cc608.ends_sentence("and then")
