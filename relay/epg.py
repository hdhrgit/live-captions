"""Programme guide: merge public XMLTV files into /epg.xml for the enabled channels.

Guide files are chosen per channel by its iptv-org country (epg.json), fetched
in the background every EPG_HOURS, and matched to channels by name (or an
alias for guides that use another script). The output uses each channel's
playlist tvg-id as the XMLTV channel id, so Channels DVR maps it directly.
"""
import calendar
import gzip
import io
import json
import logging
import os
import re
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape, quoteattr

log = logging.getLogger("epg")
HOURS = float(os.environ.get("EPG_HOURS", "8"))
DAYS = float(os.environ.get("EPG_DAYS", "3"))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "epg.json")) as _f:
    CONF = json.load(_f)

_ARAB = {"eg", "sa", "ae", "qa", "iq", "lb", "sy", "jo", "kw", "bh", "om", "ye", "ly", "dz", "ma", "tn", "sd", "ps"}


def norm(s):
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s.lower())
    s = re.sub(r"\b(hd|sd|fhd|uhd|4k|tv|channel|television|the|live)\b", " ", s)
    return re.sub(r"[^\w]+", "", s).replace("_", "")


def stem(tvg_id):
    return (tvg_id or "").split("@")[0].split(".")[0]


def split_camel(s):
    return re.sub(r"(?<=[a-z])(?=[A-Z0-9])|(?<=[0-9])(?=[A-Za-z])", " ", s)


def guide_url(name):
    return name if "://" in name else CONF["base"].format(name)


class Guide:
    def __init__(self):
        self.lock = threading.Lock()
        self.files = {}      # name -> {"channels": {id: [names]}, "programmes": {id: [xml]}, "at": t}
        self.xml = b'<?xml version="1.0" encoding="UTF-8"?>\n<tv generator-info-name="captions-dvr"></tv>\n'
        self.matches = {}    # our channel id -> "FILE:guide-id" or None
        self.updated = None
        self.wake = threading.Event()
        self.xml_translated = False
        self.translator = Translator()

    # ------------------------------------------------------------ fetch ---

    def fetch(self, name):
        url = guide_url(name)
        req = urllib.request.Request(url, headers={"User-Agent": "captions-dvr"})
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
        if raw[:2] != b"\x1f\x8b":
            raw = gzip.compress(raw)
        # Index only: channel names and programme counts. The file stays
        # compressed in memory and is re-read for the matched channels.
        chans, progs = {}, {}
        for _, el in ET.iterparse(gzip.GzipFile(fileobj=io.BytesIO(raw)), events=("end",)):
            if el.tag == "channel":
                chans[el.get("id")] = [d.text or "" for d in el.findall("display-name")]
            elif el.tag == "programme":
                progs[el.get("channel")] = progs.get(el.get("channel"), 0) + 1
            else:
                continue   # a channel's display-names are read when the channel ends
            el.clear()
        log.info("guide %s: %d channels, %d with programmes", name, len(chans), len(progs))
        return {"channels": chans, "programmes": progs, "raw": raw, "at": time.time()}

    def programmes(self, fname, gids):
        """{guide id: [programme elements]} within the guide window."""
        horizon, now, out = time.time() + DAYS * 86400, time.time(), {}
        raw = self.files[fname]["raw"]
        for _, el in ET.iterparse(gzip.GzipFile(fileobj=io.BytesIO(raw)), events=("end",)):
            if el.tag == "programme" and el.get("channel") in gids:
                stop = xmltv_time(el.get("stop") or el.get("start"))
                if stop and now - 3600 < stop < horizon:
                    out.setdefault(el.get("channel"), []).append(el)
                    continue
            if el.tag in ("programme", "channel"):
                el.clear()
        return out

    # ------------------------------------------------------------ match ---

    def match(self, ch, countries):
        """Best (file, guide channel id) for a playlist channel, or None."""
        country = countries[0] if countries else None
        tvg = (ch.get("tvg") or {}).get("tvg-id", "")
        feed = tvg.partition("@")[2]
        if feed.upper() in ("", "SD", "HD", "FHD", "UHD", "4K"):
            feed = ""
        # A named feed (France24.fr@Arabic) must match with its feed name,
        # or it would pick up the main channel's schedule.
        keys = {norm(ch["name"]), norm(split_camel(stem(tvg)) + " " + feed)} - {""}
        keys |= {norm(a) for a in CONF["aliases"].get(stem(tvg), [])}
        best = None
        for rank, fname in enumerate(self.files_for(countries)):
            g = self.files.get(fname)
            if not g:
                continue
            for gid, names in g["channels"].items():
                if not g["programmes"].get(gid):
                    continue
                by_id = norm(gid.rsplit(".", 1)[0].replace(".", " "))
                gkeys = {norm(n) for n in names} | {by_id}
                if keys & gkeys:
                    suffix = gid.rsplit(".", 1)[-1].lower()
                    # A guide id that matches itself beats one matched by a
                    # display name only (e.g. "rossia1" over "rossia1-pl1").
                    score = (suffix == country, suffix in _ARAB and country in _ARAB, -rank,
                             by_id in keys, g["programmes"][gid])
                    if best is None or score > best[0]:
                        best = (score, fname, gid)
        return (best[1], best[2]) if best else None

    @staticmethod
    def files_for(countries):
        out = []
        for cc in countries or ():
            for f in CONF["countries"].get(cc, []):
                if f not in out:
                    out.append(f)
        return out

    # ------------------------------------------------------------ build ---

    def refresh(self, channels):
        """channels: [(channel dict, [countries])] for the enabled channels."""
        wanted = []
        for _, ccs in channels:
            for f in self.files_for(ccs):
                if f not in wanted:
                    wanted.append(f)
        for f in wanted:
            g = self.files.get(f)
            if g and time.time() - g["at"] < HOURS * 3600:
                continue
            try:
                self.files[f] = self.fetch(f)
            except Exception as e:  # noqa: BLE001 - keep the previous copy
                log.warning("guide %s failed: %s", f, e)
        for f in list(self.files):
            if f not in wanted:
                del self.files[f]
        out = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<tv generator-info-name="captions-dvr">']
        body, matches, byfile = [], {}, {}
        for ch, cc in channels:
            m = self.match(ch, cc)
            matches[ch["id"]] = f"{m[0]}:{m[1]}" if m else None
            xid = (ch.get("tvg") or {}).get("tvg-id") or ch["id"]
            out.append(f"<channel id={quoteattr(xid)}><display-name>{escape(ch['name'])}"
                       f"</display-name></channel>")
            if m:
                byfile.setdefault(m[0], []).append((m[1], xid))
        langs = {((ch.get("tvg") or {}).get("tvg-id") or ch["id"]): guide_language(ch, ccs)
                 for ch, ccs in channels}
        elements = []
        for fname, pairs in byfile.items():
            progs = self.programmes(fname, {gid for gid, _ in pairs})
            for gid, xid in pairs:
                for el in progs.get(gid, []):
                    el.set("channel", xid)
                    elements.append(el)
        if not self.xml_translated:
            # Nothing translated yet: publish the original-language guide now.
            self.publish(out, [ET.tostring(el, encoding="unicode") for el in elements], matches)
        if TRANSLATE():
            t0 = time.monotonic()
            n = self.translator.translate_programmes(elements, langs)
            if n is not None:
                log.info("epg: translated %d new strings in %.0fs", n, time.monotonic() - t0)
                self.xml_translated = True
        body = [ET.tostring(el, encoding="unicode") for el in elements]
        self.publish(out, body, matches)

    def publish(self, out, body, matches):
        with self.lock:
            self.xml = "\n".join(out + body + ["</tv>", ""]).encode()
            self.matches = matches
            self.updated = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        log.info("epg: %d channels, %d with guide data, %d programmes",
                 len(matches), sum(1 for v in matches.values() if v), len(body))

    def loop(self, get_channels):
        while True:
            try:
                self.refresh(get_channels())
            except Exception as e:  # noqa: BLE001
                log.warning("epg refresh failed: %s", e)
            self.wake.wait(HOURS * 3600)
            self.wake.clear()


def xmltv_time(s):
    m = re.match(r"(\d{14})\s*([+-]\d{4})?", s or "")
    if not m:
        return None
    t = calendar.timegm(time.strptime(m.group(1), "%Y%m%d%H%M%S"))
    if m.group(2):
        sign = 1 if m.group(2)[0] == "+" else -1
        t -= sign * (int(m.group(2)[1:3]) * 3600 + int(m.group(2)[3:5]) * 60)
    return t


# ------------------------------------------------------------ translate ---

# EPG_TRANSLATE=1: translate the guide into English with NLLB on this
# machine's CPU (off by default); relay.py downloads the model if it's missing.
TRANSLATE = lambda: os.environ.get("EPG_TRANSLATE", "0") == "1"  # noqa: E731
MODEL_MISSING = lambda: None  # noqa: E731
CACHE = os.environ.get("EPG_CACHE", os.path.join(
    os.path.dirname(os.environ.get("CHANNELS_FILE", "/config/channels.json")), "epg_translations.sqlite"))

# Seconds the translation model stays loaded after a job (it holds about
# 1.4 GB, and translating leaves another gigabyte or so of freed but kept
# memory, which release_memory hands back).
NLLB_KEEP = float(os.environ.get("EPG_NLLB_KEEP", "60"))


def release_memory():
    """Return freed heap to the system (glibc keeps it per thread otherwise)."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# Default guide language by channel country, when the channel's language is "auto".
COUNTRY_LANG = {"il": "he", "tr": "tr", "ir": "fa", "kp": "ko", "kr": "ko", "jp": "ja", "cn": "zh",
                "tw": "zh", "hk": "zh", "ru": "ru", "ua": "uk", "de": "de", "fr": "fr", "it": "it",
                "es": "es", "pt": "pt", "br": "pt", "nl": "nl", "pl": "pl", "gr": "el", "ro": "ro"}
COUNTRY_LANG.update({cc: "ar" for cc in _ARAB})

_SCRIPTS = [(re.compile(r"[֐-׿]"), "he"), (re.compile(r"[가-힣]"), "ko"),
            (re.compile(r"[぀-ヿ]"), "ja"), (re.compile(r"[一-鿿]"), "zh"),
            (re.compile(r"[Ѐ-ӿ]"), "ru"), (re.compile(r"[฀-๿]"), "th"),
            (re.compile(r"[Ͱ-Ͽ]"), "el"), (re.compile(r"[ऀ-ॿ]"), "hi")]
_ARABIC = re.compile(r"[؀-ۿ]")


def guide_language(ch, countries):
    """The language a channel's guide is probably written in, or None."""
    if not ch.get("translate_guide", True):
        return None
    lang = ch.get("language") or "auto"
    if lang == "auto":
        lang = next((COUNTRY_LANG[c] for c in countries or () if c in COUNTRY_LANG), None)
    return lang


def non_latin(text):
    """Whether text has a script text_language() tells by itself."""
    return any(pat.search(text) for pat, _ in _SCRIPTS) or bool(_ARABIC.search(text))


def text_language(text, hint):
    """Language of one guide string: by script, else the channel's hint for
    Latin-script languages (e.g. Turkish). None means leave it alone."""
    for pat, lang in _SCRIPTS:
        if pat.search(text):
            return lang
    if _ARABIC.search(text):
        return "fa" if hint == "fa" else "ar"
    if hint in _LATIN_LANGS and re.search(r"[A-Za-z]", text) and not looks_english(text):
        return hint
    return None


# Latin-script languages whose guide text we translate (others: only non-Latin scripts).
_LATIN_LANGS = {"tr", "de", "fr", "es", "it", "pt", "nl", "pl", "ro", "cs", "hu", "sv", "fi", "id", "vi"}
_EN_WORDS = set("the of and to in a is for with on from by at an this that news live your you "
                "are be as it his her their its new how what who will".split())


def looks_english(text):
    words = re.findall(r"[a-z']+", text.lower())
    return bool(words) and sum(w in _EN_WORDS for w in words) / len(words) >= 0.15


def sentences(text, limit=250):
    """Split text at sentence ends (and, failing that, commas or spaces) into
    pieces short enough for NLLB."""
    parts = [p for p in re.split(r"(?<=[.!?؟۔:;])\s+", text.strip()) if p]
    out = []
    for p in parts:
        while len(p) > limit:
            cut = max(p.rfind(",", 0, limit), p.rfind("،", 0, limit), p.rfind(" ", 0, limit))
            cut = cut if cut > limit // 3 else limit
            out.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            out.append(p)
    return out or [text]


def tidy(text):
    """Undo tokenizer spacing: 'The news .' -> 'The news.', "I 'm" -> "I'm"."""
    text = re.sub(r"\s+([.,!?;:%)])", r"\1", text)
    text = re.sub(r"\s+'(s|m|re|ve|ll|d|t)\b", r"'\1", text)
    return re.sub(r"([(])\s+", r"\1", text).strip()


class Translator:
    """NLLB-200 on this machine's CPU with a persistent sqlite cache."""

    def __init__(self):
        self.db = None
        self.mt = None    # NLLB while loaded; False: not available
        self.used = 0.0   # monotonic time of the last job
        self.lock = threading.Lock()   # the managed guide and /api/epg share the model and cache

    def _ready(self):
        """Whether translation is available: the model is installed (else its
        download starts) and the cache is open. The model itself is loaded
        only when a guide has strings the cache doesn't have (_load)."""
        import nllbget
        if self.mt is False and nllbget.installed():
            self.mt = None   # downloaded since (web page or captions-dvr-get-nllb)
        if self.mt is None and not nllbget.installed():
            MODEL_MISSING()   # downloads it in the background; translated from the next refresh
            self.mt = False
        if self.mt is False:
            return False
        if self.db is None:
            import asr
            import sqlite3
            self.codes = asr.NLLB_CODES
            self.db = sqlite3.connect(CACHE, check_same_thread=False)
            # tr2: whole texts translated sentence by sentence (tr cut long ones)
            self.db.execute("CREATE TABLE IF NOT EXISTS tr2 (lang TEXT, src TEXT, en TEXT, "
                            "PRIMARY KEY (lang, src))")
        return True

    def _load(self):
        """The model, loaded for new strings (about 3 GB while it
        translates). It is unloaded EPG_NLLB_KEEP seconds after its last job."""
        if self.mt is None:
            try:
                import asr
                import nllbget
                self.mt = asr.NLLB(nllbget.default_path(), int(os.environ.get("EPG_NLLB_THREADS", "2")))
            except Exception as e:  # noqa: BLE001 - no NLLB here: guide stays untranslated
                log.warning("guide translation unavailable: %s", e)
                self.mt = False
                return False
            threading.Thread(target=self._unload_when_idle, name="epg-unload", daemon=True).start()
        self.used = time.monotonic()
        return bool(self.mt)

    def _unload_when_idle(self):
        while True:
            time.sleep(max(1.0, self.used + NLLB_KEEP - time.monotonic()))
            with self.lock:
                if time.monotonic() - self.used < NLLB_KEEP:
                    continue
                if self.mt:
                    self.mt.close()
                    self.mt = None
                    release_memory()
                    log.info("epg: translation model unloaded")
                return

    def translate_programmes(self, elements, langs):
        """Add English title/sub-title/desc before the originals. Returns the
        number of newly translated strings, or None if translation is off."""
        with self.lock:
            try:
                return self._translate_programmes(elements, langs)
            finally:
                self.used = time.monotonic()   # (the model stays for EPG_NLLB_KEEP)

    def _translate_programmes(self, elements, langs):
        if not self._ready():
            return None
        todo = []   # (programme, tag, original element, lang, text)
        for el in elements:
            hint = langs.get(el.get("channel"))
            if hint is None:
                continue
            for tag in ("title", "sub-title", "desc", "category"):
                for orig in el.findall(tag):
                    if not (orig.text or "").strip():
                        continue
                    # A guide's lang="en" holds for Latin text only: some
                    # (CCTV-13's) label Chinese titles "en".
                    if orig.get("lang") == "en" and not non_latin(orig.text):
                        continue
                    lang = text_language(orig.text, hint)
                    if lang and lang in self.codes and lang != "en":
                        todo.append((el, tag, orig, lang, orig.text.strip()))
        found, new = {}, {}
        for _, _, _, lang, text in todo:
            key = (lang, text)
            if key in found or key in new:
                continue
            row = self.db.execute("SELECT en FROM tr2 WHERE lang=? AND src=?", key).fetchone()
            if row:
                found[key] = row[0]
            else:
                new[key] = None
        if new and not self._load():
            return None
        by_lang = {}
        for lang, text in new:
            by_lang.setdefault(lang, []).append(text)
        for lang, texts in by_lang.items():
            for i in range(0, len(texts), 8):
                batch = texts[i:i + 8]
                # NLLB truncates long inputs: translate sentence by sentence.
                parts = [sentences(t) for t in batch]
                flat = [p for ps in parts for p in ps]
                out = []
                for j in range(0, len(flat), 24):
                    out += self.mt.translate(flat[j:j + 24], lang)
                k = 0
                for src, ps in zip(batch, parts):
                    new[(lang, src)] = tidy(" ".join(x.strip() for x in out[k:k + len(ps)]))
                    k += len(ps)
                self.db.executemany("INSERT OR REPLACE INTO tr2 VALUES (?, ?, ?)",
                                    [(lang, t, new[(lang, t)]) for t in batch])
                self.db.commit()
        found.update(new)
        originals = {}
        for el, tag, orig, lang, text in todo:
            en = found.get((lang, text))
            if not en:
                continue
            # English only: some clients show the original element instead.
            if tag == "title":
                originals[id(el)] = (el, orig.text.strip())
            orig.text = en
            orig.set("lang", "en")
        for el, title in originals.values():
            # Keep the original title readable at the end of the description.
            desc = el.find("desc")
            if desc is None:
                desc = ET.SubElement(el, "desc", {"lang": "en"})
                desc.text = ""
            desc.text = (desc.text or "").rstrip() + ("\n\n" if desc.text else "") + f"Original title: {title}"
        return len(new)


# ------------------------------------------------- stateless guides ---

# Common words of the Latin-script languages a FAST guide is mislabelled
# "en" in (Samsung's German or French lineups label everything English).
_STOPWORDS = {
    "de": set("der die das und mit von für ist im den dem nicht ein eine einer auf zu des sich als auch bei".split()),
    "fr": set("le la les des et du un une pour avec dans est au aux en sur par qui que ce son sa ses".split()),
    "es": set("el la los las y de del con para una un en por es que se su sus al lo como".split()),
    "it": set("il lo la gli le e di del della con per una un che non è nel nella dei degli al si".split()),
    "pt": set("o a os as e do da dos das com para uma um em não que no na se ao".split()),
}


def guess_latin(text):
    """The Latin-script language a guide string is written in, or None."""
    words = re.findall(r"[a-zàâäçéèêëîïôöùûüÿñœæßáíóú']+", text.lower())
    if len(words) < 3:
        return None
    best, score = None, 0.0
    for lang, stop in _STOPWORDS.items():
        n = sum(w in stop for w in words) / len(words)
        if n > score:
            best, score = lang, n
    return best if score >= 0.15 else None


def translate_xmltv(body, translator, default=""):
    """Translate any XMLTV guide into English (stateless mode, /api/epg).
    Each channel's language comes from its strings' lang attributes, with
    strings labelled "en" that read as German, French... relabelled first;
    non-Latin scripts are recognised whatever the label. Returns (xml bytes,
    newly translated strings or None if translation is unavailable)."""
    root = ET.fromstring(body)
    elements = root.findall("programme")
    votes = {}
    for el in elements:
        for tag in ("title", "sub-title", "desc"):
            for e in el.findall(tag):
                text = (e.text or "").strip()
                if not text:
                    continue
                lang = e.get("lang") or ""
                if lang in ("", "en") and not non_latin(text) and not looks_english(text):
                    g = guess_latin(text)
                    if g:
                        e.set("lang", g)
                        lang = g
                if lang and lang != "en":
                    votes.setdefault(el.get("channel"), {}).setdefault(lang, 0)
                    votes[el.get("channel")][lang] += 1
    langs = {ch: max(v, key=v.get) for ch, v in votes.items()}
    for el in elements:
        langs.setdefault(el.get("channel"), default)   # "": non-Latin scripts only
    n = translator.translate_programmes(elements, langs)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), n


class AdhocGuides:
    """/api/epg?url=: an outside XMLTV guide, translated. The first request
    gets the original guide straight away and starts the translation in the
    background; later ones get the translated copy. It is re-fetched and
    re-translated (only new strings: the cache keeps the rest) after
    EPG_ADHOC_MINUTES (default 60): FAST guides often cover only the next
    few hours, so an older copy runs dry."""

    TTL = float(os.environ.get("EPG_ADHOC_MINUTES", "60")) * 60

    def __init__(self, translator):
        self.translator = translator
        self.lock = threading.Lock()
        self.cache = {}   # (url, lang) -> {"xml", "at", "translated", "busy"}

    def get(self, url, lang=""):
        key = (url, lang)
        with self.lock:
            hit = self.cache.get(key)
            if hit and (hit["busy"] or time.time() - hit["at"] < self.TTL):
                return hit["xml"]
        raw = self._fetch(url)
        with self.lock:
            hit = self.cache.setdefault(key, {"xml": raw, "at": 0, "translated": False, "busy": False})
            if not hit["translated"]:
                hit["xml"] = raw
            hit["busy"] = True
        threading.Thread(target=self._translate, args=(key, raw), daemon=True).start()
        return hit["xml"]

    @staticmethod
    def _fetch(url):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data

    def _translate(self, key, raw):
        t0 = time.monotonic()
        try:
            xml, n = translate_xmltv(raw, self.translator, key[1])
        except Exception as e:  # noqa: BLE001 - keep serving the original
            log.warning("epg %s: translation failed: %s", key[0], e)
            xml, n = raw, None
        with self.lock:
            hit = self.cache[key]
            hit.update(at=time.time(), busy=False)
            if n is not None:
                hit.update(xml=xml, translated=True)
        log.info("epg %s: %s new strings translated in %.0fs", key[0],
                 "no" if n is None else n, time.monotonic() - t0)
