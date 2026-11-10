"""Download the NLLB-200 model the relay uses to translate the programme
guide (epg.py), already converted for CTranslate2 (int8, 1.4 GB).

The relay downloads it by itself when EPG_TRANSLATE=1 and it's missing; to
fetch it beforehand:  python nllbget.py [dest]   (the .deb: captions-dvr-get-nllb)

The model comes from OpenNMT (the CTranslate2 authors), its tokenizer from
Meta's own repository; both at a fixed revision and checked by size and
SHA-256. Downloads resume where they stopped. Licence: CC-BY-NC 4.0.
"""
import hashlib
import logging
import os
import sys
import threading
import time
import urllib.request

log = logging.getLogger("epg")
NAME = "nllb-200-distilled-1.3B-ct2"
CT2 = "https://huggingface.co/OpenNMT/nllb-200-distilled-1.3B-ct2-int8/resolve/70f572adafa4794890ce7826156a4209717855af/"
META = "https://huggingface.co/facebook/nllb-200-distilled-1.3B/resolve/7be3e24664b38ce1cac29b8aeed6911aa0cf0576/"
FILES = [   # (url, name, size, sha256 or None)
    (CT2 + "config.json", "config.json", 1071, None),
    (CT2 + "shared_vocabulary.json", "shared_vocabulary.json", 5921176, None),
    (META + "sentencepiece.bpe.model", "sentencepiece.bpe.model", 4852054,
     "14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a"),
    (CT2 + "model.bin", "model.bin", 1377755738,
     "89dc4b9eb7f4dcd3ff36023c9b1c55fcecb47ddc0df97ab31b068ef9cf194940"),
]
TOTAL = sum(f[2] for f in FILES)


def default_path():
    """NLLB_MODEL, else models/ next to the channel config if the model is
    there (the .deb: /var/lib/captions-dvr), else /models (Docker)."""
    if os.environ.get("NLLB_MODEL"):
        return os.environ["NLLB_MODEL"]
    here = os.path.join(os.path.dirname(os.path.abspath(os.environ.get("CHANNELS_FILE", "channels.json"))),
                        "models", NAME)
    if os.path.exists(os.path.join(here, "model.bin")) or not os.path.isdir("/models"):
        return here
    return os.path.join("/models", NAME)


def installed(path=None):
    return os.path.exists(os.path.join(path or default_path(), "model.bin"))


class Download:
    """One download at a time, in the background (the relay's guide), with
    retries; .state goes into /status as "epg_model"."""

    RETRIES = (30, 120, 600)   # seconds before each retry

    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"state": "ready" if installed() else "missing", "mb_total": round(TOTAL / 1e6)}

    def start(self, done_cb=None):
        with self.lock:
            if self.state["state"] in ("downloading", "verifying", "waiting"):
                return False
            self.state = {"state": "downloading", "percent": 0, "mb_done": 0, "mb_total": round(TOTAL / 1e6)}
        threading.Thread(target=self.run, args=(done_cb,), name="nllb-download", daemon=True).start()
        return True

    def run(self, done_cb):
        dest = default_path()
        log.info("guide translator: downloading NLLB-200 (%.1f GB) to %s", TOTAL / 1e9, dest)
        t0, logged = time.monotonic(), [-5, 0.0]

        def progress(done, name):
            pct = done * 100 // TOTAL
            self.state.update(state="downloading", file=name, percent=pct, mb_done=round(done / 1e6))
            now = time.monotonic()
            if pct >= logged[0] + 5 or now - logged[1] >= 30:
                logged[:] = [pct, now]
                log.info("guide translator: %d%% (%s, %.0f of %.0f MB, %.1f MB/s)", pct, name, done / 1e6,
                         TOTAL / 1e6, done / 1e6 / max(1.0, now - t0))

        def phase(what, name):
            self.state.update(state=what, file=name)
            if what == "verifying":
                log.info("guide translator: checking %s", name)

        for attempt in range(len(self.RETRIES) + 1):
            try:
                fetch(dest, progress, phase)
                self.state = {"state": "ready", "percent": 100, "mb_done": round(TOTAL / 1e6),
                              "mb_total": round(TOTAL / 1e6)}
                log.info("guide translator: ready (%s, %.0f s)", dest, time.monotonic() - t0)
                if done_cb:
                    done_cb()
                return
            except Exception as e:  # noqa: BLE001 - retried, then reported in /status
                err = f"{type(e).__name__}: {e}"
                if attempt == len(self.RETRIES):
                    self.state.update(state="failed", error=err, next_retry=None)
                    log.warning("guide translator: download failed (%s); will retry at the next guide "
                                "refresh, or run captions-dvr-get-nllb now", err)
                    return
                wait = self.RETRIES[attempt]
                self.state.update(state="waiting", error=err,
                                  next_retry=time.strftime("%H:%M:%S UTC", time.gmtime(time.time() + wait)))
                log.warning("guide translator: %s; retrying in %d s, resuming at %.0f MB", err, wait,
                            partial_bytes(dest) / 1e6)
                time.sleep(wait)


def partial_bytes(dest):
    """Bytes already on disk (complete files and .part), where a retry resumes."""
    n = 0
    for _, name, size, _ in FILES:
        for p in (os.path.join(dest, name), os.path.join(dest, name) + ".part"):
            if os.path.exists(p):
                n += min(size, os.path.getsize(p))
                break
    return n


def fetch(dest, progress=lambda done, name: None, phase=lambda what, name: None):
    """Download (resuming) and verify every file into dest; the model only
    counts as installed once model.bin is complete and checked."""
    os.makedirs(dest, exist_ok=True)
    done = 0
    for url, name, size, sha in FILES:
        path = os.path.join(dest, name)
        part = path + ".part"
        if os.path.exists(path) and os.path.getsize(path) == size:
            done += size
            progress(done, name)
            continue
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have < size:
            phase("downloading", name)
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
                if r.status != 206:
                    have = 0
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    have += len(block)
                    progress(done + have, name)
        if os.path.getsize(part) != size:
            raise RuntimeError(f"{name}: got {os.path.getsize(part)} bytes, expected {size}")
        if sha:
            phase("verifying", name)
            h = hashlib.sha256()
            with open(part, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != sha:
                os.remove(part)
                log.warning("guide translator: %s checksum mismatch, deleted", name)
                raise RuntimeError(f"{name}: checksum mismatch (deleted; downloaded again)")
        os.replace(part, path)
        done += size
        progress(done, name)


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else default_path()
    if installed(dest):
        sys.exit(f"already installed: {dest}")
    print(f"downloading the guide translator (NLLB-200, {TOTAL / 1e9:.1f} GB) to {dest}")
    t0 = time.monotonic()

    def bar(done, name):
        pct = done / TOTAL
        print(f"\r[{'#' * int(pct * 30):<30}] {pct * 100:5.1f}%  {done / 1e6:6.0f}/{TOTAL / 1e6:.0f} MB  "
              f"{done / 1e6 / max(1.0, time.monotonic() - t0):5.1f} MB/s  {name:<24}", end="", flush=True)
    fetch(dest, bar, lambda what, name: what == "verifying" and print(f"\nchecking {name}...", end="", flush=True))
    print("\ndone. Set EPG_TRANSLATE=1 in the relay's env file and restart it to translate the guide.")
