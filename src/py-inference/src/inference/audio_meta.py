"""Duration of uploaded audio, read from headers and container metadata, without decoding.

This is the worker's copy of the gateway's measurement (sp-state-agent
internal/gateway/audioduration*.go). Both sides bill transcription from this number, so
the two implementations must agree byte for byte on every file:

    WAV         PCM data-chunk size over the byte rate
    FLAC        STREAMINFO total samples over the sample rate
    MP3         sum of samples-per-frame / rate over the MPEG frames, an initial
                Xing/Info/VBRI frame skipped, LAME gapless delay + padding removed
    Ogg         last granule of the first Opus / Vorbis stream (Opus: minus pre-skip, 48 kHz)
    WebM / MKV  last audio block timestamp + its Opus packet length - DiscardPadding
                - CodecDelay; the Info Duration only when there are no audio blocks
    MP4 / M4A   media-header duration of the first sound track

A ValueError means the file is not one of these, or declares no length.
"""

from __future__ import annotations

import struct

UNSUPPORTED = "unsupported audio format: upload WAV, FLAC, MP3, OGG (Opus/Vorbis), WebM or M4A/MP4"


def audio_duration(b: bytes) -> tuple[float, str]:
    """Return (seconds, format)."""
    if len(b) >= 12 and b[0:4] == b"RIFF" and b[8:12] == b"WAVE":
        return _wav(b), "wav"
    if len(b) >= 4 and b[0:4] == b"fLaC":
        return _flac(b), "flac"
    if len(b) >= 4 and b[0:4] == b"OggS":
        return _ogg(b), "ogg"
    if len(b) >= 4 and b[0:4] == b"\x1a\x45\xdf\xa3":
        return _matroska(b), "webm"
    if len(b) >= 8 and b[4:8] == b"ftyp":
        return _mp4(b), "mp4"
    if (len(b) >= 3 and b[0:3] == b"ID3") or (len(b) >= 4 and _mp3_frame(b[0:4]) is not None):
        return _mp3(b), "mp3"
    raise ValueError(UNSUPPORTED)


# ---- WAV / FLAC ----

def _wav(b: bytes) -> float:
    byte_rate = 0
    have_fmt = False
    off = 12
    while off + 8 <= len(b):
        cid = b[off:off + 4]
        size = struct.unpack("<I", b[off + 4:off + 8])[0]
        body = off + 8
        if cid == b"fmt ":
            if size < 16 or body + 16 > len(b):
                raise ValueError("malformed WAV fmt chunk")
            fmt = struct.unpack("<H", b[body:body + 2])[0]
            if fmt == 0xFFFE and size >= 26 and body + 26 <= len(b):
                fmt = struct.unpack("<H", b[body + 24:body + 26])[0]
            if fmt not in (1, 3):
                raise ValueError(f"WAV encoding {fmt} is compressed; upload PCM WAV or FLAC")
            byte_rate = struct.unpack("<I", b[body + 8:body + 12])[0]
            have_fmt = True
        elif cid == b"data":
            if not have_fmt or byte_rate == 0:
                raise ValueError("WAV data chunk before a valid fmt chunk")
            n = size
            if n == 0 or body + n > len(b):
                n = len(b) - body
            return n / byte_rate
        off = body + size + size % 2
    raise ValueError("WAV has no data chunk")


def _flac(b: bytes) -> float:
    if len(b) < 4 + 4 + 34:
        raise ValueError("FLAC too short")
    if b[4] & 0x7F != 0:
        raise ValueError("FLAC STREAMINFO block missing")
    si = b[8:8 + 34]
    rate = si[10] << 12 | si[11] << 4 | si[12] >> 4
    total = (si[13] & 0x0F) << 32 | struct.unpack(">I", si[14:18])[0]
    if rate == 0:
        raise ValueError("FLAC sample rate is zero")
    if total == 0:
        raise ValueError("FLAC does not declare its length; re-encode with a known sample count")
    return total / rate


# ---- MP3 ----

_BR_V1 = [
    [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0],
    [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0],
    [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],
]
_BR_V2 = [
    [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
]
_SR = {0: (11025, 12000, 8000), 2: (22050, 24000, 16000), 3: (44100, 48000, 32000)}


def _mp3_frame(h: bytes):
    """(length, samples, rate) of one MPEG audio frame header, or None."""
    if len(h) < 4 or h[0] != 0xFF or h[1] & 0xE0 != 0xE0:
        return None
    version = (h[1] >> 3) & 3
    layer = (h[1] >> 1) & 3
    br_idx = h[2] >> 4
    sr_idx = (h[2] >> 2) & 3
    pad = (h[2] >> 1) & 1
    if version == 1 or layer == 0 or br_idx in (0, 15) or sr_idx == 3:
        return None
    rate = _SR[version][sr_idx]
    li = 3 - layer
    br = (_BR_V1 if version == 3 else _BR_V2)[li][br_idx] * 1000
    if li == 0:
        return (12 * br // rate + pad) * 4, 384, rate
    if li == 1 or version == 3:
        return 144 * br // rate + pad, 1152, rate
    return 72 * br // rate + pad, 576, rate


def _mp3_resync(b: bytes, start: int) -> int:
    for i in range(start, len(b) - 3):
        f = _mp3_frame(b[i:i + 4])
        if f is None or f[0] < 4:
            continue
        j = i + f[0]
        if j + 4 <= len(b) and _mp3_frame(b[j:j + 4]) is None:
            continue
        return i
    return -1


def _mp3_is_info(f: bytes) -> bool:
    head = f[:64]
    return b"Xing" in head or b"Info" in head or b"VBRI" in head


def _mp3_trim(f: bytes) -> int:
    i = f.find(b"Xing")
    if i < 0:
        i = f.find(b"Info")
    if i < 0 or i + 8 > len(f):
        return 0
    flags = struct.unpack(">I", f[i + 4:i + 8])[0]
    p = i + 8
    for bit, n in ((1, 4), (2, 4), (4, 100), (8, 4)):
        if flags & bit:
            p += n
    if p + 24 > len(f) or f[p:p + 4] not in (b"LAME", b"Lavf", b"Lavc"):
        return 0
    d = f[p + 21:p + 24]
    return (d[0] << 4 | d[1] >> 4) + ((d[1] & 0x0F) << 8 | d[2])


def _mp3(b: bytes) -> float:
    off = 0
    if len(b) >= 10 and b[0:3] == b"ID3":
        off = 10 + ((b[6] & 0x7F) << 21 | (b[7] & 0x7F) << 14 | (b[8] & 0x7F) << 7 | (b[9] & 0x7F))
        if b[5] & 0x10:
            off += 10
    dur = 0.0
    frames = 0
    trim = rate0 = 0
    while off + 4 <= len(b):
        f = _mp3_frame(b[off:off + 4])
        if f is None or f[0] < 4:
            tail = b[off:off + 8]
            if tail.startswith(b"TAG") or tail.startswith(b"APETAGEX") or tail.startswith(b"LYRICS"):
                break
            nxt = _mp3_resync(b, off + 1)
            if nxt < 0:
                break
            off = nxt
            continue
        length, samples, rate = f
        if frames == 0 and _mp3_is_info(b[off:min(off + length, len(b))]):
            trim = _mp3_trim(b[off:min(off + length, len(b))])
            rate0 = rate
            off += length
            frames = -1
            continue
        if frames < 0:
            frames = 0
        dur += samples / rate
        frames += 1
        off += length
    if frames <= 0:
        raise ValueError("MP3 has no audio frames")
    if trim > 0 and rate0 > 0 and trim / rate0 < dur:
        dur -= trim / rate0
    return dur


# ---- Ogg ----

def _ogg(b: bytes) -> float:
    serial = None
    codec = ""
    rate = 0.0
    pre_skip = 0
    last = -1
    off = 0
    while off + 27 <= len(b):
        if b[off:off + 4] != b"OggS":
            raise ValueError("malformed Ogg page")
        granule = struct.unpack("<q", b[off + 6:off + 14])[0]
        ser = struct.unpack("<I", b[off + 14:off + 18])[0]
        nseg = b[off + 26]
        if off + 27 + nseg > len(b):
            raise ValueError("truncated Ogg page")
        body = sum(b[off + 27:off + 27 + nseg])
        start = off + 27 + nseg
        payload = b[start:min(start + body, len(b))]
        if not codec:
            if len(payload) >= 19 and payload[0:8] == b"OpusHead":
                codec, serial, rate = "opus", ser, 48000.0
                pre_skip = struct.unpack("<H", payload[10:12])[0]
            elif len(payload) >= 16 and payload[0:7] == b"\x01vorbis":
                codec, serial = "vorbis", ser
                rate = float(struct.unpack("<I", payload[12:16])[0])
        elif ser == serial and granule >= 0:
            last = granule
        off = start + body
    if not codec:
        raise ValueError("Ogg file has no Opus or Vorbis stream")
    if last < 0 or rate <= 0:
        raise ValueError("Ogg stream declares no length")
    samples = last - pre_skip
    if samples <= 0:
        raise ValueError("Ogg stream is empty")
    return samples / rate


# ---- WebM / Matroska ----

def _ebml_id(b: bytes, off: int):
    if off >= len(b) or b[off] == 0:
        return None
    n, m = 1, 0x80
    while n <= 4 and not b[off] & m:
        n += 1
        m >>= 1
    if n > 4 or off + n > len(b):
        return None
    return int.from_bytes(b[off:off + n], "big"), n


def _ebml_size(b: bytes, off: int):
    if off >= len(b) or b[off] == 0:
        return None
    n, m = 1, 0x80
    while n <= 8 and not b[off] & m:
        n += 1
        m >>= 1
    if off + n > len(b):
        return None
    first = b[off] & (0xFF >> n)
    v = first
    all_ones = first == (0xFF >> n)
    for i in range(1, n):
        v = v << 8 | b[off + i]
        if b[off + i] != 0xFF:
            all_ones = False
    return (-1 if all_ones else v), n


def _opus_packet_samples(p: bytes) -> int:
    if not p:
        return 0
    cfg = p[0] >> 3
    if cfg < 12:
        frame = (480, 960, 1920, 2880)[cfg % 4]
    elif cfg < 16:
        frame = (480, 960)[cfg % 2]
    else:
        frame = (120, 240, 480, 960)[cfg % 4]
    c = p[0] & 3
    if c == 0:
        return frame
    if c in (1, 2):
        return 2 * frame
    return (p[1] & 0x3F) * frame if len(p) >= 2 else 0


def _matroska(b: bytes) -> float:
    ENTER = {0x18538067, 0x1549A966, 0x1654AE6B, 0xAE, 0x1F43B675, 0xA0}
    scale = 1000000
    info_dur = 0.0
    tracks: list[dict] = []
    audio_track = 0
    cluster_tc = 0
    last_tick = -1
    last_frame = 0
    codec_delay = 0
    last_discard = 0
    group_discard = 0
    off = 0
    while off < len(b):
        r = _ebml_id(b, off)
        if r is None:
            break
        eid, il = r
        r = _ebml_size(b, off + il)
        if r is None:
            break
        size, sl = r
        start = off + il + sl
        if eid in ENTER:
            if eid == 0xAE:
                tracks.append({"num": 0, "typ": 0, "codec": ""})
            off = start
            continue
        if size < 0 or start + size > len(b):
            if eid == 0x1A45DFA3:
                raise ValueError("malformed WebM header")
            break
        p = b[start:start + size]
        if eid == 0x2AD7B1:
            scale = int.from_bytes(p, "big")
        elif eid == 0x4489:
            if len(p) == 4:
                info_dur = struct.unpack(">f", p)[0]
            elif len(p) == 8:
                info_dur = struct.unpack(">d", p)[0]
        elif eid == 0xD7 and tracks:
            tracks[-1]["num"] = int.from_bytes(p, "big")
        elif eid == 0x83 and tracks:
            tracks[-1]["typ"] = int.from_bytes(p, "big")
        elif eid == 0x86 and tracks:
            tracks[-1]["codec"] = p.decode("latin-1")
        elif eid == 0x56AA and tracks and tracks[-1]["typ"] != 1:
            codec_delay = int.from_bytes(p, "big")
        elif eid == 0x75A2:
            v = int.from_bytes(p, "big")
            if 0 < len(p) < 8 and p[0] & 0x80:
                v -= 1 << (8 * len(p))
            group_discard = v
            last_discard = v
        elif eid == 0xE7:
            cluster_tc = int.from_bytes(p, "big")
        elif eid in (0xA3, 0xA1):
            if audio_track == 0:
                for t in tracks:
                    if t["typ"] == 2:
                        audio_track = t["num"]
                        break
            r = _ebml_size(p, 0)
            if r is not None:
                tn, tl = r
                if tn >= 0 and len(p) >= tl + 3 and tn == audio_track and audio_track != 0:
                    rel = struct.unpack(">h", p[tl:tl + 2])[0]
                    tick = cluster_tc + rel
                    if tick >= last_tick:
                        last_tick = tick
                        last_frame = 0
                        last_discard = group_discard
                        group_discard = 0
                        if p[tl + 2] & 0x06 == 0:
                            last_frame = _opus_packet_samples(p[tl + 3:])
        off = start + size
    is_opus = any(t["num"] == audio_track and t["codec"] == "A_OPUS" for t in tracks)
    if last_tick >= 0:
        d = last_tick * scale / 1e9
        if is_opus:
            d += last_frame / 48000 - last_discard / 1e9 - codec_delay / 1e9
        if d <= 0:
            raise ValueError("WebM audio is empty")
        return d
    if info_dur > 0:
        return info_dur * scale / 1e9
    raise ValueError("WebM file has no audio blocks")


# ---- MP4 ----

def _mp4_children(b: bytes, frm: int, to: int):
    out = []
    off = frm
    while off + 8 <= to:
        size = struct.unpack(">I", b[off:off + 4])[0]
        typ = b[off + 4:off + 8]
        hdr = 8
        if size == 1:
            if off + 16 > to:
                break
            size = struct.unpack(">Q", b[off + 8:off + 16])[0]
            hdr = 16
        elif size == 0:
            size = to - off
        if size < hdr or off + size > to:
            break
        out.append((typ, off + hdr, off + size))
        off += size
    return out


def _mp4(b: bytes) -> float:
    def find(boxes, typ):
        for bx in boxes:
            if bx[0] == typ:
                return bx
        return None
    moov = find(_mp4_children(b, 0, len(b)), b"moov")
    if moov is None:
        raise ValueError("MP4 has no moov box (upload a complete file)")
    for trak in _mp4_children(b, moov[1], moov[2]):
        if trak[0] != b"trak":
            continue
        mdia = find(_mp4_children(b, trak[1], trak[2]), b"mdia")
        if mdia is None:
            continue
        mb = _mp4_children(b, mdia[1], mdia[2])
        hdlr, mdhd = find(mb, b"hdlr"), find(mb, b"mdhd")
        if hdlr is None or mdhd is None or hdlr[2] - hdlr[1] < 12 or b[hdlr[1] + 8:hdlr[1] + 12] != b"soun":
            continue
        p = b[mdhd[1]:mdhd[2]]
        if len(p) >= 20 and p[0] == 0:
            scale, dur = struct.unpack(">II", p[12:20])
            if dur == 0xFFFFFFFF:
                dur = 0
        elif len(p) >= 32 and p[0] == 1:
            scale = struct.unpack(">I", p[20:24])[0]
            dur = struct.unpack(">Q", p[24:32])[0]
            if dur == 0xFFFFFFFFFFFFFFFF:
                dur = 0
        else:
            raise ValueError("malformed MP4 mdhd box")
        if scale == 0 or dur == 0:
            raise ValueError("MP4 sound track declares no length (fragmented MP4 is not supported)")
        return dur / scale
    raise ValueError("MP4 has no sound track")
