"""deploy/version.sh, and a new install's empty config."""
import os
import subprocess

import config

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
                   check=True, capture_output=True)


def version(repo, *args):
    os.makedirs(repo / "deploy", exist_ok=True)
    script = repo / "deploy" / "version.sh"
    script.write_text(open(os.path.join(ROOT, "deploy", "version.sh")).read())
    return subprocess.run(["sh", str(script), *args], capture_output=True, text=True, check=True).stdout.strip()


def test_version_on_and_after_tag(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "f").write_text("1")
    git(tmp_path, "add", "f")
    git(tmp_path, "commit", "-qm", "one")
    assert version(tmp_path).startswith("0.0.0+g")
    git(tmp_path, "tag", "v0.1.0")
    assert version(tmp_path) == "0.1.0"
    (tmp_path / "f").write_text("2")
    git(tmp_path, "commit", "-qam", "two")
    v = version(tmp_path, "3")
    assert v.startswith("0.1.0+r3.g") and len(v) > len("0.1.0+r3.g")


def test_new_install_is_empty(tmp_path):
    c = config.Config(str(tmp_path / "channels.json"))
    assert c.channels == [] and c.sources == []
