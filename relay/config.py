"""Channel sources (M3U playlists) and the persisted channel selection.

The config file (CHANNELS_FILE, JSON) holds:
  {"sources": [{"id", "name", "url", "added"}],
   "channels": [{"id", "source", "name", "url", "number", "enabled",
                 "language", "headers": {"referer", "user_agent"},
                 "tvg": {...}, "live_start_index"?}]}
Nothing from a new source is enabled; the user picks channels in the web UI.
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.request

IPTV_ORG = "https://iptv-org.github.io/iptv/countries/{}.m3u"
# The raw per-country stream files list more feeds (e.g. Al Jazeera, Sky News
# Arabia) than the generated country playlists; a country source merges both.
IPTV_ORG_RAW = "https://raw.githubusercontent.com/iptv-org/iptv/master/streams/{}.m3u"
FIRST_NUMBER = 8501

_ATTR = re.compile(r'([\w-]+)="([^"]*)"')
lock = threading.RLock()

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "majors.json")) as _f:
    _MAJ = json.load(_f)
MAJORS, MAJOR_IDS = {}, {}
for _cc, _pats in _MAJ.items():
    if _cc.startswith("_"):
        continue
    _names = [p for p in _pats if not p.startswith("id:")]
    MAJOR_IDS[_cc] = {p[3:].lower() for p in _pats if p.startswith("id:")}
    if _names:
        MAJORS[_cc] = re.compile(r"(?<![\w])(?:" + "|".join(_names) + r")", re.I)
EXTRA = _MAJ.get("_extra", {})
LANGUAGE = _MAJ.get("_language", {})


def stem(tvg_id):
    return (tvg_id or "").split("@")[0].split(".")[0]


def channel_countries(c, sources=()):
    """Countries for a channel: its tvg-id suffix, then its iptv-org source's."""
    out = []
    m = re.match(r"[^.@]+\.([a-z]{2})\b", (c.get("tvg") or {}).get("tvg-id", ""))
    if m:
        out.append(m.group(1))
    src = next((s for s in sources if s["id"] == c.get("source")), None)
    cc = country_of(src) if src else None
    if cc and cc not in out:
        out.append(cc)
    return out


def country_of(source):
    """The iptv-org country code of a source, or None for other playlists."""
    m = re.search(r"/countries/([a-z]{2})\.m3u", source.get("url", ""))
    return m.group(1) if m else None


def is_major(name, country, tvg_id=""):
    if stem(tvg_id).lower() in MAJOR_IDS.get(country or "", ()):
        return True
    pat = MAJORS.get(country or "")
    return bool(pat and pat.search(name))


def source_url(spec):
    """A two-letter country code means that iptv-org country playlist."""
    spec = spec.strip()
    if re.fullmatch(r"[A-Za-z]{2}", spec):
        return IPTV_ORG.format(spec.lower())
    if not re.match(r"https?://", spec):
        raise ValueError("expected a 2-letter country code or an http(s) M3U URL")
    return spec


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def split_extinf(line):
    """(attributes, title): the title follows the first comma outside quotes."""
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif ch == "," and not quoted:
            return line[:i], line[i + 1:]
    return line, ""


def parse_m3u(text):
    """Entries {name, url, tvg, headers} from an extended M3U playlist."""
    out, cur = [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXTINF"):
            head, title = split_extinf(line)
            attrs = dict(_ATTR.findall(head))
            headers = {}
            if attrs.get("http-referrer"):
                headers["referer"] = attrs.pop("http-referrer")
            if attrs.get("http-user-agent"):
                headers["user_agent"] = attrs.pop("http-user-agent")
            cur = {"name": title.strip(), "tvg": attrs, "headers": headers}
        elif line.startswith("#EXTVLCOPT:") and cur is not None:
            key, _, val = line[len("#EXTVLCOPT:"):].partition("=")
            if key == "http-referrer":
                cur["headers"]["referer"] = val
            elif key == "http-user-agent":
                cur["headers"]["user_agent"] = val
        elif line and not line.startswith("#"):
            entry = cur or {"name": line, "tvg": {}, "headers": {}}
            entry["url"] = line
            out.append(entry)
            cur = None
    return out


def channel_id(name, url):
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "ch"
    return f"{slug}-{hashlib.sha1(url.encode()).hexdigest()[:6]}"


class Config:
    def __init__(self, path):
        self.path = path
        self.sources, self.channels = [], []
        self.settings = {}   # relay-wide: compare_mts, burn_height (updated in place)
        self.load()

    def load(self):
        with lock:
            if not os.path.exists(self.path):
                return
            with open(self.path) as f:
                data = json.load(f)
            if isinstance(data, list):  # old format: a plain list of channels
                data = {"sources": [], "channels": [
                    dict(c, url=c["source"], source="manual", enabled=True,
                         language=c.get("language", "ko")) for c in data]}
            self.sources, self.channels = data.get("sources", []), data.get("channels", [])
            self.settings.clear()
            self.settings.update(data.get("settings", {}))
            self.mark_majors()
            for c in self.channels:
                c.setdefault("headers", {})
                c.setdefault("tvg", {})
                c.setdefault("language", "auto")
                c.setdefault("enabled", False)
                if c.get("captions") == "auto":   # the old default: now "generate"
                    c["captions"] = "generate"

    def save(self):
        with lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"sources": self.sources, "channels": self.channels, "settings": self.settings}, f,
                          ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)

    def mark_majors(self):
        """Flag channels whose name matches the country's major networks."""
        countries = {s["id"]: country_of(s) for s in self.sources}
        extras = {cc: {x.split(":", 1)[1].split("@")[0].lower() for x in xs} for cc, xs in EXTRA.items()}
        for c in self.channels:
            cc = countries.get(c["source"])
            tvg = c.get("tvg", {}).get("tvg-id", "")
            c["major"] = (c["source"] == "manual" or is_major(c["name"], cc, tvg)
                          or stem(tvg).lower() in extras.get(cc, ()))

    def feed_urls(self, c):
        """c's URL first (unless known broken), then working feeds of the same channel."""
        tid = (c.get("tvg") or {}).get("tvg-id")
        own = c["url"]
        alts = []
        if tid:
            with lock:
                alts = [x for x in self.channels if x is not c and x["url"] != own
                        and (x.get("tvg") or {}).get("tvg-id") == tid
                        and (x.get("health") or {}).get("ok")]
            alts.sort(key=lambda x: -((x.get("health") or {}).get("height") or 0))
        urls = [x["url"] for x in alts]
        if (c.get("health") or {}).get("ok", True):
            return [own] + urls
        return urls + [own] if urls else [own]

    def get(self, cid):
        return next((c for c in self.channels if c["id"] == cid), None)

    def enabled(self):
        return sorted((c for c in self.channels if c.get("enabled")),
                      key=lambda c: int(c.get("number") or 0))

    def next_number(self):
        used = {int(c["number"]) for c in self.channels if c.get("number")}
        n = FIRST_NUMBER
        while n in used:
            n += 1
        return n

    def add_source(self, spec, name=None):
        url = source_url(spec)
        entries = parse_m3u(fetch(url))
        cc = re.search(r"/countries/([a-z]{2})\.m3u", url)
        if cc:
            try:
                entries += parse_m3u(fetch(IPTV_ORG_RAW.format(cc.group(1))))
            except Exception:  # noqa: BLE001 - the generated list alone still works
                pass
        for extra in EXTRA.get(cc.group(1) if cc else "", []):
            # Channels listed under another country that belong with this one.
            other, _, want = extra.partition(":")
            want_stem, _, want_feed = want.partition("@")
            try:
                for e in parse_m3u(fetch(IPTV_ORG_RAW.format(other))):
                    tid = e["tvg"].get("tvg-id", "")
                    if stem(tid) == want_stem and (not want_feed or tid.endswith("@" + want_feed)):
                        entries.append(e)
            except Exception:  # noqa: BLE001 - extras are optional
                pass
        with lock:
            src = next((s for s in self.sources if s["url"] == url), None)
            if src is None:
                sid = re.sub(r"[^a-z0-9]+", "-", (name or url.rsplit("/", 1)[-1]).lower()).strip("-")
                src = {"id": sid or f"src{len(self.sources) + 1}", "name": name or spec.strip(),
                       "url": url}
                self.sources.append(src)
            src["updated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            known = {c["url"]: c for c in self.channels}
            added = 0
            for e in entries:
                if e["url"] in known:
                    # Refresh what the playlist says; keep the user's choices.
                    c = known[e["url"]]
                    if c["source"] == src["id"]:
                        c.update(name=e["name"], tvg=e["tvg"], headers=e["headers"])
                    continue
                known[e["url"]] = {"source": None}  # duplicate URL later in the list: skip
                self.channels.append({"id": channel_id(e["name"], e["url"]), "source": src["id"],
                                      "name": e["name"], "url": e["url"], "number": None,
                                      "enabled": False,
                                      "language": LANGUAGE.get(
                                          re.sub(r"\.[a-z]{2}(?=@|$)", lambda m: m.group(0), e["tvg"].get("tvg-id", ""))
                                      ) or LANGUAGE.get(stem(e["tvg"].get("tvg-id")), "auto"),
                                      "headers": e["headers"], "tvg": e["tvg"]})
                added += 1
            src["count"] = sum(1 for c in self.channels if c["source"] == src["id"])
            self.mark_majors()
            src["majors"] = sum(1 for c in self.channels if c["source"] == src["id"] and c["major"])
            self.save()
            return src, added

    def remove_source(self, sid):
        """Drops the source and its channels that are not enabled."""
        with lock:
            self.sources = [s for s in self.sources if s["id"] != sid]
            self.channels = [c for c in self.channels if c["source"] != sid or c.get("enabled")]
            self.save()

    def update_settings(self, changes):
        """Compare defaults: which backends compare mode shows, and its resolution."""
        with lock:
            if "compare_mts" in changes:
                mts = [m for m in ("seamless", "nllb", "whisper") if m in (changes["compare_mts"] or ())]
                if not mts:
                    raise ValueError("compare_mts needs at least one of seamless, nllb, whisper")
                self.settings["compare_mts"] = mts
            if "burn_height" in changes:
                if int(changes["burn_height"]) not in (360, 480, 540, 720, 1080):
                    raise ValueError("burn_height must be 360, 480, 540, 720 or 1080")
                self.settings["burn_height"] = int(changes["burn_height"])
            if "asr_parallel" in changes:
                n = int(changes["asr_parallel"])
                if not 1 <= n <= 32:
                    raise ValueError("asr_parallel must be 1 to 32")
                self.settings["asr_parallel"] = n
            if "archive" in changes:
                self.settings["archive"] = bool(changes["archive"])
            if "audio_ffmpeg" in changes:
                path = str(changes["audio_ffmpeg"] or "").strip()
                if path and not os.access(path, os.X_OK):
                    raise ValueError(f"{path} is not an executable file")
                self.settings["audio_ffmpeg"] = path
            self.save()
            return self.settings

    def update_channel(self, cid, changes):
        with lock:
            c = self.get(cid)
            if c is None:
                raise KeyError(cid)
            if "enabled" in changes:
                c["enabled"] = bool(changes["enabled"])
                if c["enabled"] and not c.get("number"):
                    c["number"] = self.next_number()
            if "number" in changes:
                n = str(changes["number"]).strip()
                c["number"] = int(n) if n.isdigit() else None
            if "language" in changes:
                lang = str(changes["language"]).strip().lower() or "auto"
                if not re.fullmatch(r"auto|[a-z]{2,3}", lang):
                    raise ValueError("language must be auto or an ISO code like en, ko")
                c["language"] = lang
            if "translate_guide" in changes:
                c["translate_guide"] = bool(changes["translate_guide"])
            if "compare_channels" in changes:
                c["compare_channels"] = bool(changes["compare_channels"])
            if "compare" in changes:
                c["compare"] = bool(changes["compare"])
            if "long_chunks" in changes:
                c["long_chunks"] = bool(changes["long_chunks"])
            if "compare_mts" in changes:
                mts = [m for m in ("seamless", "nllb", "whisper") if m in (changes["compare_mts"] or ())]
                if not mts:
                    raise ValueError("compare_mts needs at least one of seamless, nllb, whisper")
                c["compare_mts"] = mts
            if "burn_height" in changes:
                if int(changes["burn_height"]) not in (360, 480, 540, 720, 1080):
                    raise ValueError("burn_height must be 360, 480, 540, 720 or 1080")
                c["burn_height"] = int(changes["burn_height"])
            if "sentences" in changes:
                c["sentences"] = bool(changes["sentences"])
            if "eia" in changes:
                c["eia"] = bool(changes["eia"])
            if "vtt_chars" in changes:
                c["vtt_chars"] = int(changes["vtt_chars"])
            if "vtt_lines" in changes:
                c["vtt_lines"] = max(1, min(3, int(changes["vtt_lines"])))
            if "cc_layout" in changes:
                if changes["cc_layout"] not in ("balanced", "fill", "single"):
                    raise ValueError("cc_layout must be balanced, fill or single")
                c["cc_layout"] = changes["cc_layout"]
            if "cc_center" in changes:
                if changes["cc_center"] not in ("nbtsp", "pad", "visible", "underscore", "underscore_all", "native",
                                                "608only", "608space", "608tsp"):
                    raise ValueError("cc_center must be nbtsp, pad, visible, underscore, underscore_all, native, "
                                     "608only, 608space or 608tsp")
                c["cc_center"] = changes["cc_center"]
            if "cc_position" in changes:
                if changes["cc_position"] not in ("bottom", "middle", "top"):
                    raise ValueError("cc_position must be bottom, middle or top")
                c["cc_position"] = changes["cc_position"]
            if "mt" in changes:
                if changes["mt"] not in ("nllb", "whisper", "seamless"):
                    raise ValueError("mt must be seamless, nllb or whisper")
                c["mt"] = changes["mt"]
            if "captions" in changes:
                if changes["captions"] not in ("source", "generate"):
                    raise ValueError("captions must be generate or source")
                c["captions"] = changes["captions"]
            if "name" in changes and str(changes["name"]).strip():
                c["name"] = str(changes["name"]).strip()
            self.save()
            return c
