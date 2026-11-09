"""Dissect A/53 closed captions in a TS: 608 control codes, 708 windows, PMT descriptors.

Usage: python ccdump.py file.ts [seconds]
"""
import collections
import sys

import av

PATH = sys.argv[1]
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 60


def pmt_descriptors(path):
    """caption_service_descriptor (0x86) and others on the video ES in the PMT."""
    data = open(path, "rb").read(188 * 20000)
    pmt_pid, out = None, []
    for i in range(0, len(data) - 188, 188):
        p = data[i:i + 188]
        if p[0] != 0x47 or not p[1] & 0x40:
            continue
        pid = (p[1] & 0x1F) << 8 | p[2]
        off = 4 + (p[4] + 1 if p[3] & 0x20 else 0)
        sec = p[off + 1 + p[off]:]
        if pid == 0 and pmt_pid is None:
            n = ((sec[1] & 0x0F) << 8 | sec[2]) - 9
            for j in range(8, 8 + n, 4):
                prog = sec[j] << 8 | sec[j + 1]
                if prog:
                    pmt_pid = (sec[j + 2] & 0x1F) << 8 | sec[j + 3]
                    break
        elif pid == pmt_pid:
            slen = (sec[1] & 0x0F) << 8 | sec[2]
            pil = (sec[10] & 0x0F) << 8 | sec[11]
            j = 12 + pil
            while j < 3 + slen - 4:
                st, epid, eil = sec[j], (sec[j + 1] & 0x1F) << 8 | sec[j + 2], (sec[j + 3] & 0x0F) << 8 | sec[j + 4]
                descs = sec[j + 5:j + 5 + eil]
                k = 0
                while k < len(descs):
                    tag, ln = descs[k], descs[k + 1]
                    out.append((hex(st), hex(epid), hex(tag), descs[k + 2:k + 2 + ln].hex()))
                    k += 2 + ln
                j += 5 + eil
            return out
    return out


CTRL = {0x20: "RCL", 0x21: "BS", 0x24: "DER", 0x25: "RU2", 0x26: "RU3", 0x27: "RU4", 0x28: "FON",
        0x29: "RDC", 0x2A: "TR", 0x2B: "RTD", 0x2C: "EDM", 0x2D: "CR", 0x2E: "ENM", 0x2F: "EOC"}
PAC_ROW = {(0x11, 0): 1, (0x11, 1): 2, (0x12, 0): 3, (0x12, 1): 4, (0x15, 0): 5, (0x15, 1): 6, (0x16, 0): 7,
           (0x16, 1): 8, (0x17, 0): 9, (0x17, 1): 10, (0x10, 0): 11, (0x13, 0): 12, (0x13, 1): 13,
           (0x14, 0): 14, (0x14, 1): 15}

stats = collections.Counter()
cc_counts, rows, cmds, text = collections.Counter(), collections.Counter(), collections.Counter(), []
windows, wattrs, pens = [], [], []
pkt708 = bytearray()


def parse_708(pkt):
    """Service blocks in one DTVCC packet; record DefineWindow / SetWindowAttributes."""
    i = 1
    while i < len(pkt):
        svc, size = pkt[i] >> 5, pkt[i] & 0x1F
        i += 1
        if svc == 7:
            svc = pkt[i] & 0x3F
            i += 1
        if size == 0:
            break
        blk = pkt[i:i + size]
        i += size
        stats[f"708 service {svc}"] += 1
        j = 0
        while j < len(blk):
            c = blk[j]
            if 0x98 <= c <= 0x9F and j + 6 < len(blk):
                b = blk[j + 1:j + 7]
                windows.append(dict(id=c - 0x98, visible=bool(b[0] & 0x20), rel=bool(b[1] & 0x80),
                                    av=b[1] & 0x7F, ah=b[2], anchor=b[3] >> 4, rows=(b[3] & 0x0F) + 1,
                                    cols=(b[4] & 0x3F) + 1, wstyle=(b[5] >> 3) & 7, pstyle=b[5] & 7))
                j += 7
            elif c == 0x97 and j + 4 < len(blk):
                wattrs.append(dict(justify=blk[j + 3] & 3, print_dir=(blk[j + 3] >> 4) & 3,
                                   display_effect=blk[j + 4] & 3))
                j += 5
            elif c == 0x92 and j + 2 < len(blk):
                pens.append((blk[j + 1] & 0x0F, blk[j + 2] & 0x3F))
                j += 3
            elif 0x20 <= c <= 0x7F:
                stats["708 text bytes"] += 1
                j += 1
            else:
                stats[f"708 cmd 0x{c:02x}"] += 1
                j += 1


with av.open(PATH) as c:
    v = c.streams.video[0]
    first = None
    for f in c.decode(v):
        first = f.time if first is None else first
        if f.time - first > SECS:
            break
        sd = [d for d in f.side_data if "A53" in str(d.type).upper() or "CLOSED" in str(d.type).upper()]
        if not sd:
            stats["frames without cc"] += 1
            continue
        b = bytes(sd[0])
        stats["frames with cc"] += 1
        cc_counts[len(b) // 3] += 1
        for k in range(0, len(b) - 2, 3):
            hdr, d1, d2 = b[k], b[k + 1] & 0x7F, b[k + 2] & 0x7F
            valid, typ = bool(hdr & 4), hdr & 3
            if not valid:
                stats[f"type{typ} invalid/padding"] += 1
                continue
            stats[f"type{typ} valid"] += 1
            if typ in (0, 1):
                fld = "f1" if typ == 0 else "f2"
                if d1 == 0 and d2 == 0:
                    stats[f"{fld} null"] += 1
                elif 0x10 <= d1 <= 0x1F:
                    ch = 2 if d1 & 0x08 else 1
                    d1c = d1 & ~0x08
                    if d1c in (0x14, 0x15) and 0x20 <= d2 <= 0x2F:
                        cmds[f"{fld} CC{ch} {CTRL.get(d2, hex(d2))}"] += 1
                    elif d1c == 0x17 and 0x21 <= d2 <= 0x23:
                        cmds[f"{fld} CC{ch} TO{d2 - 0x20}"] += 1
                    elif d2 >= 0x40:
                        row = PAC_ROW.get((d1c, (d2 & 0x20) >> 5))
                        indent = ((d2 & 0x0E) >> 1) * 4 if d2 & 0x10 else None
                        rows[f"{fld} CC{ch} row {row} " + (f"indent {indent}" if indent is not None else f"style {(d2 & 0x0E) >> 1}")] += 1
                    elif d1c == 0x11 and 0x20 <= d2 <= 0x2F:
                        cmds[f"{fld} CC{ch} midrow/attr"] += 1
                    else:
                        cmds[f"{fld} other ctrl"] += 1
                elif d1 >= 0x20:
                    if typ == 0 and len(text) < 400:
                        text.append(chr(d1) + (chr(d2) if d2 >= 0x20 else ""))
                    stats[f"{fld} text"] += 1
            elif typ == 3:
                if pkt708:
                    parse_708(pkt708)
                pkt708 = bytearray([b[k + 1], b[k + 2]])
            elif typ == 2:
                pkt708 += bytes([b[k + 1], b[k + 2]])

print("== PMT descriptors (stream_type, pid, tag, data):", pmt_descriptors(PATH))
print("== cc_count per frame:", dict(cc_counts))
print("== counts:", dict(stats))
print("== 608 commands:", dict(cmds.most_common(20)))
print("== 608 PACs:", dict(rows.most_common(12)))
print("== 608 text sample:", "".join(text)[:300])
uniq = []
for w in windows:
    if w not in uniq:
        uniq.append(w)
print("== 708 DefineWindow (unique):", uniq[:8])
print("== 708 SetPenLocation (row, col):", pens[:12])
print("== 708 SetWindowAttributes (unique):", [dict(t) for t in {tuple(sorted(x.items())) for x in wattrs}][:6])
