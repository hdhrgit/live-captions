"""Decoding a source's own CEA-608 captions (CC1) to text.

When a channel passes its source captions through ("CC: auto"), their text
also goes into the WebVTT track, so the subtitle track is never empty. The
cc_data comes straight from the video packets (A/53 "GA94" user data, in
H.264 SEI or MPEG-2 picture user data); no video decoding.
"""
import re

GA94 = re.compile(rb"GA94\x03")
# 608 characters that differ from ASCII.
CHARS = {0x2A: "á", 0x5C: "é", 0x5E: "í", 0x5F: "ó", 0x60: "ú", 0x7B: "ç", 0x7C: "÷",
         0x7D: "Ñ", 0x7E: "ñ", 0x7F: "█"}
SPECIAL = {0x30: "®", 0x31: "°", 0x32: "½", 0x33: "¿", 0x34: "™", 0x35: "¢", 0x36: "£",
           0x37: "♪", 0x38: "à", 0x39: " ", 0x3A: "è", 0x3B: "â", 0x3C: "ê", 0x3D: "î",
           0x3E: "ô", 0x3F: "û"}


def cc_pairs(data):
    """Field-1 608 byte pairs (parity stripped) in a video packet's A/53 cc_data."""
    out = []
    for m in GA94.finditer(data):
        i = m.end()
        if i + 2 > len(data):
            continue
        count = data[i] & 0x1F
        i += 2   # flags/cc_count, em_data
        for k in range(count):
            j = i + 3 * k
            if j + 3 > len(data):
                break
            head, b1, b2 = data[j], data[j + 1] & 0x7F, data[j + 2] & 0x7F
            if head & 0x04 and head & 0x03 == 0:   # cc_valid, NTSC field 1
                out.append((b1, b2))
    return out


def char(b):
    return CHARS.get(b, chr(b)) if b >= 0x20 else ""


class Decoder:
    """CC1 text. feed(pairs) returns the caption to show when it changes
    ("" when cleared), else None. Roll-up and paint-on captions grow letter
    by letter on screen; they count as changed only per completed line."""

    def __init__(self):
        self.mode = "pop"
        self.buffer = ""        # pop-on: loading, not shown
        self.shown = ""         # what's on screen
        self.lines = [""]       # roll-up
        self.rows = 2
        self.chan1 = True
        self.last_ctrl = None
        self.changed = False

    def feed(self, pairs):
        self.changed = False
        for b1, b2 in pairs:
            if b1 == 0 and b2 == 0:
                continue
            if 0x10 <= b1 <= 0x1F:
                if (b1, b2) == self.last_ctrl:   # control codes are sent twice
                    self.last_ctrl = None
                    continue
                self.last_ctrl = (b1, b2)
                self.chan1 = b1 < 0x18
                if self.chan1:
                    self.control(b1, b2)
                continue
            self.last_ctrl = None
            if self.chan1:
                self.text(char(b1) + char(b2))
        return self.shown if self.changed else None

    def text(self, s):
        if self.mode == "pop":
            self.buffer += s
        elif self.mode == "roll":
            self.lines[-1] += s
        else:
            self.shown += s

    def newline(self):
        if self.mode == "pop":
            if self.buffer and not self.buffer.endswith("\n"):
                self.buffer += "\n"
        elif self.mode == "paint" and self.shown and not self.shown.endswith("\n"):
            self.shown += "\n"

    def control(self, b1, b2):
        c = b1 & 0x07   # 0x14/0x1C -> 4, 0x11/0x19 -> 1, ...
        if c == 4 and 0x20 <= b2 <= 0x2F:
            if b2 == 0x20:
                self.mode = "pop"
            elif b2 in (0x25, 0x26, 0x27):
                self.mode, self.rows = "roll", b2 - 0x23
            elif b2 == 0x29:
                self.mode = "paint"
            elif b2 == 0x21:   # backspace
                if self.mode == "pop":
                    self.buffer = self.buffer[:-1]
                else:
                    self.shown = self.shown[:-1]
            elif b2 == 0x2C:   # erase displayed memory
                self.shown, self.lines, self.changed = "", [""], True
            elif b2 == 0x2E:   # erase non-displayed memory
                self.buffer = ""
            elif b2 == 0x2F:   # end of caption: show the loaded one
                self.shown, self.buffer, self.changed = self.buffer.strip(), "", True
            elif b2 == 0x2D and self.mode == "roll":   # carriage return: a line is complete
                if self.lines[-1].strip():
                    self.shown, self.changed = self.lines[-1].strip(), True
                self.lines = (self.lines + [""])[-(self.rows + 1):]
            elif b2 == 0x2D and self.mode == "paint" and self.shown.strip():
                self.changed = True
        elif c == 1 and 0x30 <= b2 <= 0x3F:
            self.text(SPECIAL.get(b2, ""))
        elif c == 1 and 0x20 <= b2 <= 0x2F:   # mid-row code: a space
            self.text(" ")
        elif c in (2, 3) and 0x20 <= b2 <= 0x3F:   # extended character: replaces the previous
            if self.mode == "pop":
                self.buffer = self.buffer[:-1]
        elif b2 >= 0x40:   # preamble address code: a new row
            self.newline()
