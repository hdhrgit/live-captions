"""Shared test setup: the relay's modules on the path, and an environment
that never reaches real services (no ASR, config in a temp dir)."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "relay"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_tmp = tempfile.mkdtemp(prefix="captions-dvr-test-")
os.environ.setdefault("CHANNELS_FILE", os.path.join(_tmp, "channels.json"))
os.environ.setdefault("ASR_BACKEND", "none")
os.environ.pop("ASR_URL", None)
os.environ.pop("EPG_TRANSLATE", None)


def pytest_terminal_summary(terminalreporter):
    """The source tests' summary table (tests/report/*.json), at the end of the run."""
    import json
    import glob
    report = os.environ.get("TEST_REPORT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "report"))
    files = sorted(glob.glob(os.path.join(report, "*.json")))
    if not files:
        return
    w = terminalreporter.write_line
    lines = ["", "| Test | Format | Card (ms) | First segment (s) | Segments (card) | Cues | Lead (s) | Result |",
             "|---|---|---|---|---|---|---|---|"]
    for f in files:
        test = os.path.basename(f)[:-5]
        for name, r in sorted(json.load(open(f)).items()):
            lines.append(f"| {test} | {name} | {r.get('card_ms') or '-'} | {r['first_segment_s']} | "
                         f"{r['segments']} ({r['card_segments']}) | {r['cues']} | {r['lead_s']} | "
                         f"{'OK' if not r['problems'] else 'FAIL: ' + '; '.join(r['problems'])} |")
    terminalreporter.section("source tests")
    for line in lines:
        w(line)
    with open(os.path.join(report, "summary.md"), "w") as out:
        out.write("\n".join(lines[1:]) + "\n")


def pytest_sessionstart(session):
    """A fresh report directory per run."""
    import shutil
    shutil.rmtree(os.environ.get("TEST_REPORT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")),
                  ignore_errors=True)
