"""Turning engine output into something downstream code can rely on, and the
export formats (JSON, SRT, WebVTT, plain text)."""

from __future__ import annotations

import json
import re

from .models import Segment, Transcript

_SPACE = re.compile(r"\s+")


def clean(segments: list[Segment], duration: float) -> list[Segment]:
    """Normalise segments so every consumer can make the same assumptions:

    - text is trimmed, whitespace collapsed, and no segment is empty;
    - times are clamped to [0, duration], start <= end, and ordered;
    - a segment that repeats the one before it word for word is dropped.
      That is Whisper's best-known failure, a loop on silence or music.
    - ids run 1..n with no gaps.
    """
    out: list[Segment] = []
    for s in sorted(segments, key=lambda s: s.start):
        text = _SPACE.sub(" ", s.text).strip()
        if not text:
            continue
        if out and text.casefold() == out[-1].text.casefold():
            out[-1].end = round(min(max(out[-1].end, s.end), duration), 3)  # keep the span, drop the repeat
            continue
        start = min(max(s.start, 0.0), duration)
        end = min(max(s.end, start), duration)
        out.append(Segment(0, round(start, 3), round(end, 3), text, s.confidence))
    for i, s in enumerate(out, 1):
        s.id = i
    return out


def to_json(t: Transcript) -> str:
    return json.dumps(t.to_dict(), ensure_ascii=False, indent=2)


def to_text(t: Transcript) -> str:
    return "\n".join(s.text for s in t.segments) + "\n"


def to_srt(t: Transcript) -> str:
    return "\n".join(
        f"{s.id}\n{_stamp(s.start, ',')} --> {_stamp(s.end, ',')}\n{s.text}\n" for s in t.segments
    )


def to_vtt(t: Transcript) -> str:
    body = "\n".join(f"{_stamp(s.start, '.')} --> {_stamp(s.end, '.')}\n{s.text}\n" for s in t.segments)
    return "WEBVTT\n\n" + body


FORMATS = {"json": to_json, "srt": to_srt, "vtt": to_vtt, "txt": to_text}
MEDIA_TYPES = {"json": "application/json", "srt": "application/x-subrip", "vtt": "text/vtt", "txt": "text/plain"}


def _stamp(seconds: float, sep: str) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"
