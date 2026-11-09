"""CEA-608 pop-on caption encoder and A/53 SEI builder.

The encoder hands out one CC1 byte pair per video frame, in presentation
order, so a cue's EOC (display) lands on the first frame at or after its
start time.
"""
import bisect
import re
import unicodedata

ROWS = 15
COLS = 32

# CC1 control codes (first byte, second byte)
RCL = (0x14, 0x20)  # resume caption loading (pop-on)
EDM = (0x14, 0x2C)  # erase displayed memory
ENM = (0x14, 0x2E)  # erase non-displayed memory
EOC = (0x14, 0x2F)  # end of caption (flip memories)
TAB = {1: (0x17, 0x21), 2: (0x17, 0x22), 3: (0x17, 0x23)}

# Row -> (first byte, second byte base) for CC1 preamble address codes.
PAC_ROW = {
    1: (0x11, 0x40), 2: (0x11, 0x60), 3: (0x12, 0x40), 4: (0x12, 0x60),
    5: (0x15, 0x40), 6: (0x15, 0x60), 7: (0x16, 0x40), 8: (0x16, 0x60),
    9: (0x17, 0x40), 10: (0x17, 0x60), 11: (0x10, 0x40), 12: (0x13, 0x40),
    13: (0x13, 0x60), 14: (0x14, 0x40), 15: (0x14, 0x60),
}

# ASCII characters the 608 basic set redefines, mapped to safe stand-ins.
_REMAP = {"*": "#", "\\": "/", "^": "", "_": "-", "`": "'", "{": "(",
          "}": ")", "|": "/", "~": "-", "‘": "'", "’": "'",
          "“": '"', "”": '"', "—": "-", "–": "-",
          "…": "..."}


def _parity(b):
    b &= 0x7F
    return b | (0x80 if bin(b).count("1") % 2 == 0 else 0)


PAD = (0x80, 0x80)


# Characters sent as CC1 special-character codes (each one column wide).
SPECIAL = {"\u266a": (0x11, 0x37)}


def clean_text(text):
    out = []
    for ch in text:
        if ch in SPECIAL:
            out.append(ch)
            continue
        ch = _REMAP.get(ch, ch)
        for c in unicodedata.normalize("NFKD", ch):
            if 0x20 <= ord(c) < 0x7F:
                out.append(_REMAP.get(c, c))
    return re.sub(r"\s+", " ", "".join(out)).strip()


_JOINERS = set("and but or nor so yet that which who whom whose where when while because "
               "to in on at for with from of by about as than if".split())


def balance(lines, width=COLS):
    """Re-split a two-line group so both lines are about the same length,
    preferring a break after punctuation, then before a conjunction or
    preposition; never inside a word.

    Players that left-align lines inside a centred caption block (Channels
    on tvOS) otherwise show a long first line over a short second one.
    """
    if len(lines) != 2:
        return lines
    words = " ".join(lines).split(" ")
    best, best_cost = lines, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) > width or len(b) > width:
            continue
        cost = max(len(a), len(b)) * 4
        if words[i - 1][-1:] in ",.;:?!":
            cost -= 12
        elif words[i].lower() in _JOINERS:
            cost -= 6
        if best_cost is None or cost < best_cost:
            best, best_cost = [a, b], cost
    return best


_ABBREV = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|St|Prof|Gen|Col|Lt|Sgt|Rep|Sen|Gov|vs|etc|No)|\b[A-Z](?:\.[A-Z])*)\.$")


def sentences(text):
    """text split after . ? ! (not after abbreviations like "Mr." or "U.S.")."""
    out = []
    for part in re.split(r"(?<=[.?!])\s+(?=[-\"'(A-Z0-9])", text):
        if out and _ABBREV.search(out[-1]):
            out[-1] += " " + part
        else:
            out.append(part)
    return [x for x in out if x.strip()]


def sentence_pieces(text, width=COLS, rows=2):
    """text as cue-sized pieces that follow sentences: each sentence starts a
    new piece, short sentences share one while they fit in `rows` lines, and
    a longer sentence is split at , ; : into pieces that fit (a clause that
    still doesn't fit stays whole and is wrapped over several cues)."""
    fits = lambda t: len(wrap(t, width)) <= rows  # noqa: E731
    pieces, cur = [], ""
    for sentence in sentences(text):
        joined = f"{cur} {sentence}" if cur else sentence
        if fits(joined):
            cur = joined
            continue
        if cur:
            pieces.append(cur)
        cur = ""
        if fits(sentence):
            cur = sentence
            continue
        for clause in re.split(r"(?<=[,;:])\s+", sentence):
            joined = f"{cur} {clause}" if cur else clause
            if cur and not fits(joined):
                pieces.append(cur)
                cur = clause
            else:
                cur = joined
    if cur:
        pieces.append(cur)
    return pieces


def ends_sentence(text):
    return bool(re.search(r"[.?!…][\"')\]]*$", text.strip()))


def layout_text(text, width=COLS, rows=2, layout="balanced", ascii_only=True, sentences=True):
    """Split text into cue groups of up to `rows` lines of up to `width`
    characters. width 0: no line breaks (one cue). layout "balanced" (with
    sentences) starts each sentence in a new cue and balances the lines;
    "fill" fills lines greedily; "single" forces one line per cue.
    ascii_only=False keeps any script (WebVTT tracks in the source language)."""
    clean = clean_text(text) if ascii_only else re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if not width:
        return [[clean]]
    if layout == "balanced" and sentences:
        return [g for piece in sentence_pieces(clean, width, rows) for g in _layout(piece, width, rows, layout)]
    return _layout(clean, width, rows, layout)


def _layout(clean, width, rows, layout):
    if layout == "single":
        rows = 1
    lines = wrap(clean, width)
    if layout != "fill" and len(lines) > 1:
        # Even out line lengths across the whole text: the narrowest width
        # that doesn't need more cues than greedy filling does.
        target = -(-len(lines) // rows) * rows
        for w in range(-(-len(clean) // target), width + 1):
            trial = wrap(clean, w)
            if len(trial) <= target:
                lines = trial
                break
    groups = [lines[i:i + rows] for i in range(0, len(lines), rows)]
    if layout != "fill" and rows == 2:
        groups = [balance(g, width) for g in groups]
    return groups


MIN_SHOW = 1.2      # seconds a cue stays up at least
READ_CPS = 17.0     # reading speed: characters per second


def timed_groups(start, end, groups):
    """(start, end, lines) per group, the span shared by text length. Each
    cue stays up for at least MIN_SHOW and long enough to read at READ_CPS,
    running on into the following silence; the next cue cuts it short."""
    total = sum(len(" ".join(g)) for g in groups) or 1
    t, out = start, []
    for g in groups:
        chars = len(" ".join(g))
        dur = (end - start) * chars / total
        need = 90000 * max(MIN_SHOW, chars / READ_CPS)
        out.append((int(t), int(t + max(dur, need)), g))
        t += dur
    return out


def wrap(text, width=COLS):
    lines, cur = [], ""
    for word in text.split(" "):
        while len(word) > width:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:width])
            word = word[width:]
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _pac(row, col):
    """PAC + tab offset pairs positioning the cursor at row/col."""
    hi, base = PAC_ROW[row]
    indent = (col // 4) * 4
    pairs = [(hi, base | 0x10 | (indent // 4) << 1)] * 2
    if col - indent:
        pairs += [TAB[col - indent]] * 2
    return pairs


# Where a two-line cue sits: bottom rows 14-15 (default), above a news
# lower third (rows 10-11), or at the top (rows 2-3).
POSITIONS = {"bottom": ROWS, "middle": 11, "top": 3}


def cue_pairs(lines, position="bottom"):
    """Byte pairs that load `lines` (<=2), centred, into non-displayed memory."""
    pairs = [RCL, RCL, ENM, ENM]
    last_row = POSITIONS.get(position, ROWS)
    first_row = last_row - len(lines) + 1
    for i, line in enumerate(lines):
        pairs += _pac(first_row + i, (COLS - len(line)) // 2)
        chars = []
        for c in line + "\0":
            if c in SPECIAL or c == "\0":
                if len(chars) % 2:
                    chars.append(0)
                pairs += [(chars[j], chars[j + 1]) for j in range(0, len(chars), 2)]
                chars = []
                if c in SPECIAL:
                    pairs += [SPECIAL[c]] * 2
            else:
                chars.append(ord(c))
    return pairs


class Cue:
    __slots__ = ("start", "end", "lines", "position", "pad")

    def __init__(self, start, end, lines, position="bottom", pad="nbtsp"):
        self.start, self.end, self.lines, self.position, self.pad = start, end, lines, position, pad

    def __lt__(self, other):
        return self.start < other.start


class Encoder:
    """Produces one CC1 pair per frame for frames fed in PTS order."""

    def __init__(self, frame_ticks=1800):
        self.cues = []          # sorted, not yet loaded
        self.loading = []       # pairs still to send for the loaded cue
        self.loaded = None      # cue in non-displayed memory awaiting EOC
        self.shown_end = None   # end time of the displayed cue
        self.burst = []         # control code repeats to send next
        self.frame_ticks = frame_ticks
        self.dtv = DTVCC()      # the same cues as a CEA-708 service

    def add(self, cue):
        bisect.insort(self.cues, cue)

    def split_segment(self, start, end, text, position="bottom", pad="nbtsp", layout="balanced"):
        """Turn one ASR segment into pop-on cues.

        layout "balanced": up to two rows of about equal length; "fill": first
        row filled to 32 columns; "single": one row (up to 28 columns) per cue.
        """
        groups = layout_text(text, 28 if layout == "single" else COLS, 2, layout)
        made = []
        for a, b, g in timed_groups(start, end, groups):
            cue = Cue(a, b, g, position, pad)
            self.add(cue)
            made.append(cue)
        return made

    def next_frame(self, pts):
        """(CC1 pair, 708 triplets) for the next frame in presentation order."""
        pair = self.next_pair(pts)
        return pair, self.dtv.triplets(DTV_TRIPLETS)

    def next_pair(self, pts):
        if self.burst:
            return self.burst.pop(0)
        if self.loading:
            return self.loading.pop(0)
        if self.loaded is not None and pts >= self.loaded.start:
            cue, self.loaded = self.loaded, None
            self.shown_end = cue.end
            self.burst = [EOC]
            self.dtv.show()
            return EOC
        if self.shown_end is not None and pts >= self.shown_end:
            # Keep a finished cue up if the next one follows within 0.5 s.
            nxt = self.loaded or (self.cues[0] if self.cues else None)
            if nxt is None or nxt.start - pts > 45000:
                self.shown_end = None
                self.burst = [EDM]
                self.dtv.hide()
                return EDM
        if self.loaded is None:
            while self.cues and self.cues[0].end <= pts:
                self.cues.pop(0)  # too late to show at all
            if self.cues:
                cue = self.cues[0]
                pairs = cue_pairs(cue.lines, cue.position)
                lead = (len(pairs) + 10) * self.frame_ticks
                if pts >= cue.start - lead:
                    self.cues.pop(0)
                    self.loaded = cue
                    self.loading = pairs
                    self.dtv.load(cue.lines, cue.position, cue.pad)
                    return self.loading.pop(0)
        return PAD


# ---------------------------------------------------------------- 708 ---

DTV_TRIPLETS = 9                    # 708 triplets after the one 608 triplet (10 per frame, as broadcasts)
PAD_708 = bytes([0xFA, 0x00, 0x00])  # cc_valid=0, DTVCC_PACKET_DATA
# CEA-708 C1 commands
CLW, DSW, HDW, SPA, SPC, SPL, SWA, DF0 = 0x88, 0x89, 0x8A, 0x90, 0x91, 0x92, 0x97, 0x98
# (anchor point, anchor vertical %, anchor horizontal %) per caption position
WINDOW_ANCHOR = {"bottom": (7, 90, 50), "middle": (7, 68, 50), "top": (1, 6, 50)}


class DTVCC:
    """CEA-708 service 1 mirroring the 608 pop-on cues.

    Two windows are used like 608's memories: a cue loads into the hidden
    one, then display flips them. Windows are anchored bottom- (or top-)
    centre at a relative position, centre-justified, 32 columns wide, so
    players place and centre them themselves.
    """

    def __init__(self):
        self.seq = 0
        self.out = bytearray()   # pending (cc_type, byte, byte) triplets, flattened
        self.shown = None        # window id on screen
        self.hidden = 0          # window id to load next

    def triplets(self, n):
        res = []
        while self.out and len(res) < n:
            res.append(bytes(self.out[:3]))
            del self.out[:3]
        return res

    def packet(self, atoms):
        """Queue service-1 commands as DTVCC packets.

        atoms: byte strings that must not be split (a command with its
        parameters, or text). They are packed into service blocks of at most
        31 bytes and packets of at most 128 bytes.
        """
        blocks, cur = [], b""
        for a in atoms:
            if len(cur) + len(a) > 31:
                blocks.append(cur)
                cur = b""
            cur += a
        if cur:
            blocks.append(cur)
        packets, data = [], b""
        for blk in blocks:
            blk = bytes([(1 << 5) | len(blk)]) + blk
            if 1 + len(data) + len(blk) > 127:
                packets.append(data)
                data = b""
            data += blk
        packets.append(data)
        for data in packets:
            if len(data) % 2 == 0:
                data += b"\x00"            # header + data must be an even size
            size = 1 + len(data)
            pkt = bytes([(self.seq << 6) | ((size // 2) & 0x3F)]) + data
            self.seq = (self.seq + 1) & 3
            for i in range(0, len(pkt), 2):
                self.out += bytes([0xFF if i == 0 else 0xFE]) + pkt[i:i + 2]

    def load(self, lines, position="bottom", pad="nbtsp"):
        """pad: how to centre each line for renderers that ignore justify and
        pen column. "pad": leading spaces on a transparent pen background;
        "nbtsp": 708 non-breaking transparent spaces (G2, not trimmed as
        whitespace); "visible": spaces on the black text background (a
        diagnostic); "native": window justify + pen column only; "608only":
        no 708 service at all."""
        self.off = pad == "608only"
        if self.off:
            return
        w = self.hidden
        anchor, av, ah = WINDOW_ANCHOR.get(position, WINDOW_ANCHOR["bottom"])
        rows = max(1, len(lines))
        atoms = [bytes([DF0 + w,
                        0x18,                    # hidden, row and column lock, priority 0
                        0x80 | av, ah,           # relative positioning, anchor in percent
                        (anchor << 4) | (rows - 1), COLS - 1,
                        (1 << 3) | 1])]          # window style 1, pen style 1
        # Transparent fill, scroll up; justify left when padding, else centre.
        atoms.append(bytes([SWA, 0xC0, 0x00, 0x0E if pad == "native" else 0x0C, 0x00]))
        atoms.append(bytes([SPA, 0x05, 0x03]))   # standard size, mono sans
        atoms.append(bytes([CLW, 1 << w]))
        for row, line in enumerate(lines):
            col = (COLS - len(line)) // 2
            if pad == "nbtsp":
                atoms.append(bytes([SPL, row, 0]))
                atoms += [bytes([0x10, 0x21])] * col               # EXT1 + NBTSP
            elif pad in ("underscore", "underscore_all"):
                # Pad both sides to the longest line with underscores drawn
                # black on black, so every row is as wide as the block.
                width = max(len(x) for x in lines)
                left = (width - len(line)) // 2
                right = width - len(line) - left
                atoms.append(bytes([SPL, row, (COLS - width) // 2]))
                if left:
                    atoms.append(bytes([SPC, 0x00, 0x00, 0x00]) + b"_" * left)
                atoms.append(bytes([SPC, 0x3F, 0x00, 0x00]))
                text = bytes(0x7F if ch == "\u266a" else ord(ch) for ch in line)
                if pad == "underscore_all":
                    # Inner spaces too: black underscore between white words.
                    for k, word in enumerate(text.split(b" ")):
                        if k:
                            atoms.append(bytes([SPC, 0x00, 0x00, 0x00]) + b"_" + bytes([SPC, 0x3F, 0x00, 0x00]))
                        atoms += [word[i:i + 16] for i in range(0, len(word), 16)]
                else:
                    atoms += [text[i:i + 16] for i in range(0, len(text), 16)]
                if right:
                    atoms.append(bytes([SPC, 0x00, 0x00, 0x00]) + b"_" * right)
                continue
            elif pad == "visible":
                atoms.append(bytes([SPL, row, 0]))                 # plain spaces on the text background
                atoms.append(bytes([SPC, 0x3F, 0x00, 0x00]) + b" " * col)
            elif pad != "native":
                atoms.append(bytes([SPL, row, 0]))
                if col:
                    atoms.append(bytes([SPC, 0x3F, 0xC0, 0x00]))  # white on transparent: invisible pad
                    atoms.append(b" " * col)
            else:
                # Centre by column too, for players that ignore justify.
                atoms.append(bytes([SPL, row, col]))
            atoms.append(bytes([SPC, 0x3F, 0x00, 0x00]))          # white on solid black
            text = bytes(0x7F if ch == "\u266a" else ord(ch) for ch in line)  # 0x7F: music note in G0
            atoms += [text[i:i + 16] for i in range(0, len(text), 16)]
        self.packet(atoms)

    def show(self):
        if getattr(self, "off", False):
            return
        w, old = self.hidden, self.shown
        cmd = [bytes([DSW, 1 << w])]
        if old is not None and old != w:
            cmd.append(bytes([HDW, 1 << old]))
        self.packet(cmd)
        self.shown, self.hidden = w, (old if old is not None else 1 - w)

    def hide(self):
        if getattr(self, "off", False):
            return
        self.packet([bytes([HDW, 0x03]), bytes([CLW, 0x03])])
        self.shown = None


def _escape(rbsp):
    out, zeros = bytearray(), 0
    for b in rbsp:
        if zeros >= 2 and b <= 3:
            out.append(3)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def sei_nal(pair, dtv=()):
    """H.264 SEI NAL (Annex B): one CC1 pair, then DTV_TRIPLETS 708 triplets.

    Like broadcast streams, every frame carries cc_count = 10.
    """
    # Order as in broadcast streams: the 608 field-1 pair first, then 708.
    cc = bytes([0xFC, _parity(pair[0]), _parity(pair[1])])
    dtv = list(dtv) + [PAD_708] * (DTV_TRIPLETS - len(dtv))
    cc += b"".join(dtv)
    t35 = (bytes([0xB5, 0x00, 0x31]) + b"GA94" + bytes([0x03, 0x40 | (1 + len(dtv)), 0xFF])
           + cc + b"\xFF")
    sei = bytes([0x04, len(t35)]) + t35 + b"\x80"
    return b"\x00\x00\x00\x01\x06" + _escape(sei)


_START = re.compile(b"\x00\x00\x01")


def insert_sei(data, nal):
    """Insert `nal` before the first VCL NAL of an Annex B access unit."""
    for m in _START.finditer(data):
        if m.end() < len(data) and 1 <= (data[m.end()] & 0x1F) <= 5:
            pos = m.start() - 1 if m.start() and data[m.start() - 1] == 0 else m.start()
            return data[:pos] + nal + data[pos:]
    return data


def strip_mpeg2_cc(data):
    """Remove the picture user_data carrying A/53 (GA94) captions from an
    MPEG-2 frame. Lossless: user_data never contains a start code."""
    out, pos = [], 0
    while True:
        i = data.find(b"\x00\x00\x01\xb2GA94", pos)
        if i < 0:
            out.append(data[pos:])
            return b"".join(out)
        end = data.find(b"\x00\x00\x01", i + 4)
        out.append(data[pos:i])
        pos = len(data) if end < 0 else end


def strip_cc(data, codec):
    """The frame without the source's embedded CEA-608/708 captions."""
    if b"GA94" not in data:
        return data
    if codec == "mpeg2video":
        return strip_mpeg2_cc(data)
    return strip_cc_sei(data, hevc=codec == "hevc")


def strip_cc_sei(data, hevc=False):
    """Remove SEI NALs carrying A/53 (GA94) caption data from an access unit
    (H.264 type 6; HEVC prefix SEI, type 39)."""
    starts = [m.start() for m in _START.finditer(data)]
    if not starts:
        return data
    out, prev = [], 0
    for i, st in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(data)
        nal = data[st:end]
        kind = (nal[3] >> 1) & 0x3F if hevc else nal[3] & 0x1F
        if len(nal) > 3 and kind == (39 if hevc else 6) and b"GA94" in nal:
            out.append(data[prev:st].rstrip(b"\x00") if st and data[st - 1] == 0 else data[prev:st])
            prev = end
    out.append(data[prev:])
    return b"".join(out)
