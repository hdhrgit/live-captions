"""Stateless mode: captions for any stream URL, without the channel list.

The DVR keeps its own channels and guide and points each stream at
/api/stream?url=<source>&mt=...; /api/m3u?url=<playlist> rewrites a whole
playlist that way. A stream's options are encoded in its channel id
("x-" + base64url JSON), so the HLS URLs inside the master playlist work
again after a relay restart without any stored state.

The backend (mt) is not part of the id: every backend variant of one source
shares a Channel (one pull, one ASR session). The first backend asked for
becomes its main track, others become its comparison tracks and are served
as that track's own master playlist (master-<mt>.m3u8).
"""
import base64
import ipaddress
import json
import os
import re
import socket
import threading
import time
import urllib.parse
import urllib.request

MAX_SESSIONS = int(os.environ.get("MAX_ADHOC_SESSIONS", "6"))
# Optional allow-list of source hosts: names (a suffix matches subdomains)
# and CIDRs, comma-separated. Empty allows any http(s) source.
SOURCE_ALLOW = [x.strip() for x in os.environ.get("SOURCE_ALLOW", "").split(",") if x.strip()]
M3U_CACHE_S = 300
BACKENDS = {"sml": "seamless", "seamless": "seamless", "nllb": "nllb", "wsp": "whisper", "whisper": "whisper"}
# Query parameters that /api/m3u passes on to each /api/stream URL.
# Compare mode's backends and resolution are relay settings (web page), not
# per URL.
STREAM_PARAMS = ("mt", "lang", "pos", "compare")


class Refused(Exception):
    """A request the relay won't serve; str() is the reason shown to the client."""

    def __init__(self, reason, code=400):
        super().__init__(reason)
        self.code = code


def allowed(url):
    """Raises Refused unless url is http(s) and its host passes SOURCE_ALLOW."""
    p = urllib.parse.urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise Refused("only http and https sources")
    if not SOURCE_ALLOW:
        return
    host = p.hostname.lower()
    if any(host == a or host.endswith("." + a) for a in SOURCE_ALLOW if "/" not in a):
        return
    nets = [ipaddress.ip_network(a, strict=False) for a in SOURCE_ALLOW if "/" in a]
    try:
        ips = {ipaddress.ip_address(i[4][0]) for i in socket.getaddrinfo(host, None)}
    except OSError:
        ips = set()
    if nets and any(ip in n for ip in ips for n in nets):
        return
    raise Refused(f"source host {host} is not in SOURCE_ALLOW")


def options(q):
    """Channel options from /api/stream query parameters (dict of lists)."""
    get = lambda k, d=None: (q.get(k) or [d])[0]  # noqa: E731
    url = get("url")
    if not url:
        raise Refused("missing url= parameter (the source stream, URL-encoded); got: "
                      + (", ".join(f"{k}={v[0][:60]}" for k, v in q.items()) or "nothing"))
    allowed(url)
    o = {"u": url}
    mt = BACKENDS.get((get("mt") or "").lower())
    lang = (get("lang") or "auto").lower()
    if not re.fullmatch(r"auto|[a-z]{2,3}", lang):
        raise Refused("lang must be auto or a code like he, ar")
    if lang != "auto":
        o["l"] = lang
    if get("pos") in ("middle", "top"):
        o["p"] = get("pos")
    if get("compare") in ("1", "true", "yes"):
        o["c"] = 1
    if get("name"):
        o["n"] = get("name")[:80]
    for k, name in (("referer", "r"), ("user_agent", "a")):
        if get(k):
            o[name] = get(k)
    return o, mt


_want_lock = threading.Lock()


def want(c, mt):
    """Registers backend mt on an ad-hoc channel config; returns the subtitle
    track that carries it ("main" for the channel's first backend)."""
    with _want_lock:
        c.setdefault("mt", mt)
        if mt == c["mt"]:
            return "main"
        mts = c.setdefault("compare_mts", [c["mt"]])
        if mt not in mts:
            mts.append(mt)
        c["compare"] = True
        return mt


def token(o):
    return "x-" + base64.urlsafe_b64encode(json.dumps(o, separators=(",", ":")).encode()).decode().rstrip("=")


def channel_cfg(cid):
    """The channel config encoded in an ad-hoc channel id, or None."""
    if not cid.startswith("x-"):
        return None
    try:
        b = cid[2:]
        o = json.loads(base64.urlsafe_b64decode(b + "=" * (-len(b) % 4)))
        allowed(o["u"])
    except (ValueError, KeyError, Refused):
        return None
    # Shown on screen while starting: the playlist's channel name, never the
    # source's host or IP.
    c = {"id": cid, "name": o.get("n", ""), "url": o["u"], "enabled": True, "adhoc": True,
         "language": o.get("l", "auto"), "cc_position": o.get("p", "bottom"), "burn": False,
         "headers": {k: o[n] for k, n in (("referer", "r"), ("user_agent", "a")) if o.get(n)}}
    if o.get("c"):   # burned-in compare video (backends and resolution: relay settings)
        c["compare"] = c["burn"] = True
    return c


def probe(url, headers):
    """Raises Refused if the source doesn't answer (a dead stream gets a clear error)."""
    h = {"User-Agent": headers.get("user_agent") or "Mozilla/5.0"}
    if headers.get("referer"):
        h["Referer"] = headers["referer"]
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=8) as r:
            r.read(1024)
    except Exception as e:  # noqa: BLE001 - any failure: the source is unusable now
        raise Refused(f"source not reachable: {type(e).__name__}: {e}", 502) from e


_m3u_cache, _m3u_lock = {}, threading.Lock()


def fetch_m3u(url):
    if not url:
        raise Refused("missing url= parameter (the playlist, URL-encoded), e.g. "
                      "/api/m3u?url=https%3A%2F%2Fiptv-org.github.io%2Fiptv%2Fcountries%2Fil.m3u")
    allowed(url)
    with _m3u_lock:
        hit = _m3u_cache.get(url)
        if hit and time.monotonic() - hit[0] < M3U_CACHE_S:
            return hit[1]
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        raise Refused(f"playlist not reachable: {type(e).__name__}: {e}", 502) from e
    with _m3u_lock:
        _m3u_cache[url] = (time.monotonic(), text)
    return text


LABELS = {"seamless": "Seamless", "nllb": "NLLB", "whisper": "Whisper"}


def _title_at(extinf):
    """Index of the comma before the channel title (the first one outside quotes)."""
    quoted = False
    for i, ch in enumerate(extinf):
        if ch == '"':
            quoted = not quoted
        elif ch == "," and not quoted:
            return i
    return len(extinf)


def _variant(extinf, number, label):
    """extinf with channel number `number` (if not None) and " [label]" after the name."""
    if number is not None:
        extinf = re.sub(r'\s(tvg-chno|channel-number)="[^"]*"', "", extinf)
        extinf = re.sub(r"^(#EXTINF:\s*-?\d+)", rf'\1 tvg-chno="{number}"', extinf)
    if label:
        i = _title_at(extinf)
        extinf = re.sub(r'(tvg-name="[^"]*)"', rf'\1 [{label}]"', extinf[:i]) + extinf[i:] + f" [{label}]"
    return extinf


def rewrite_m3u(text, public_url, q):
    """The playlist with each stream URL pointed at /api/stream; every other
    line (#EXTINF attributes, names, #EXTVLCOPT) unchanged.

    mt=a,b,c lists backends: each channel then appears as N (first backend),
    N.1, N.2... with " [Backend]" after the name and the same tvg-id. Channels
    without a number are numbered in playlist order; chno_offset=K adds K to
    every number (for a separate Channels source per backend)."""
    mts = [BACKENDS[m] for m in re.split(r"[,\s]+", (q.get("mt") or [""])[0].lower()) if m in BACKENDS]
    mts = list(dict.fromkeys(mts)) or [None]
    offset = int((q.get("chno_offset") or ["0"])[0] or 0)
    # chno=9000&chno_step=0.001: the whole playlist under one number,
    # 9000.001, 9000.002... (backend variants get consecutive numbers too).
    base_no = (q.get("chno") or [""])[0]
    step_s = (q.get("chno_step") or ["1"])[0] or "1"
    step, places = float(step_s), len(step_s.partition(".")[2])
    if base_no and not base_no.isdigit():
        raise Refused("chno must be a whole number, e.g. chno=9000")
    if not 0.001 <= step <= 1:
        raise Refused("chno_step must be between 0.001 and 1")
    numbered = len(mts) > 1 or offset != 0 or bool(base_no)
    extra = [(k, q[k][0]) for k in STREAM_PARAMS if q.get(k) and k != "mt"]
    out, entry, headers, index = [], [], {}, 0
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"#EXTVLCOPT:http-(referrer|user-agent)=(.*)", s, re.I)
        if m:
            headers["referer" if m.group(1).lower() == "referrer" else "user_agent"] = m.group(2).strip()
        elif s.startswith("#EXTINF"):
            headers = {}
            for k, name in (("http-referrer", "referer"), ("http-user-agent", "user_agent")):
                a = re.search(rf'{k}="([^"]*)"', s)
                if a:
                    headers[name] = a.group(1)
        if s and not s.startswith("#") and re.match(r"https?://", s, re.I):
            index += 1
            extinf = next((x for x in entry if x.strip().startswith("#EXTINF")), None)
            own = re.search(r'(?:tvg-chno|channel-number)="(\d+)(?:\.\d+)?"', extinf or "")
            base = (int(own.group(1)) if own else index) + offset
            for k, mt in enumerate(mts):
                if base_no:
                    n = (index - 1) * len(mts) + k + 1
                    if step < 1 and n * step >= 1:
                        continue   # would reach the next whole number: dropped
                    no = f"{float(base_no) + n * step:.{places}f}"
                else:
                    no = f"{base}.{k}" if k else base
                for x in entry:
                    if x is extinf and numbered:
                        x = _variant(x, no, LABELS[mt] if k and mt else "")
                    out.append(x)
                title = (extinf or "")[_title_at(extinf or ""):].lstrip(",").strip()
                params = ([("url", s)] + ([("mt", mt)] if mt else []) + extra + sorted(headers.items())
                          + ([("name", title)] if title else []))
                out.append(f"{public_url}/api/stream?" + urllib.parse.urlencode(params))
            entry, headers = [], {}
        elif s.startswith("#EXTM3U") and not out and not entry:
            out.append(line)
        elif s:
            entry.append(line)
    return "\n".join(out + entry) + "\n"
