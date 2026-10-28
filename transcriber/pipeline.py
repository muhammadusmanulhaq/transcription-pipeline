"""The pipeline: probe -> decode -> chunk -> transcribe -> re-time -> clean."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from . import audio
from .engines import Engine, EngineResult, RawSegment
from .models import Segment, Transcript
from .postprocess import clean

log = logging.getLogger(__name__)

PROMPT_CHARS = 200  # how much of the previous chunk's text is carried into the next


def transcribe_file(
    path: str | Path,
    engine: Engine,
    *,
    language: str | None = None,
    chunk_seconds: float = 300.0,
    checkpoint_dir: str | Path | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Transcript:
    """Transcribe one file of any length, with timestamps from the start of the file.

    Long files are handled in chunks:

    - Memory stays bounded: one chunk of samples at a time, never the whole file.
    - Each chunk's timestamps are shifted by where the chunk starts, so
      segments come out on the file's timeline, not the chunk's.
    - The language is detected once, on the first chunk, then pinned. A
      chunk of mostly music or noise can't then be "detected" as another language.
    - The tail of each chunk's text is passed as the next chunk's prompt,
      which keeps names, spelling and punctuation consistent across the cut.
    - With `checkpoint_dir`, every finished chunk is written to disk. A
      crash two hours into a three-hour file resumes at the chunk it died
      on, not at zero.
    """
    path = Path(path)
    info = audio.probe(path)
    checkpoint = _Checkpoint.open(checkpoint_dir, path, engine, language, chunk_seconds)
    done = checkpoint.load() if checkpoint else {}

    segments: list[Segment] = []
    detected = language
    prompt: str | None = None
    end = 0.0

    for part in audio.chunk(audio.decode(path), target_seconds=chunk_seconds):
        result = done.get(part.index)
        if result is None:
            result = engine.transcribe(part.samples, language=detected, prompt=prompt)
            if checkpoint:
                checkpoint.save(part.index, result)
        detected = detected or result.language

        for s in result.segments:
            # The engine knows only the chunk; move each segment onto the file's
            # timeline, and keep it inside its own chunk.
            start = part.start + min(max(s.start, 0.0), part.duration)
            stop = part.start + min(max(s.end, s.start), part.duration)
            segments.append(Segment(0, start, stop, s.text, s.confidence))

        text = " ".join(s.text.strip() for s in result.segments).strip()
        prompt = text[-PROMPT_CHARS:] if text else prompt
        end = part.end
        if on_progress and info.duration:
            on_progress(min(end / info.duration, 1.0))

    if on_progress:
        on_progress(1.0)
    return Transcript(
        language=detected,
        duration=end,  # counted from decoded samples: exact, unlike container headers
        engine=engine.name,
        segments=clean(segments, end),
    )


class _Checkpoint:
    """One JSON line per finished chunk, keyed by the file's content and the settings.

    The key covers anything that changes the result (the audio itself, the
    engine, the language, the chunk length), so a checkpoint is never
    reused for a different input.
    """

    def __init__(self, file: Path):
        self.file = file

    @classmethod
    def open(cls, directory, path: Path, engine: Engine, language, chunk_seconds) -> "_Checkpoint | None":
        if directory is None:
            return None
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        h.update(f"|{engine.name}|{language}|{chunk_seconds}".encode())
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return cls(directory / f"{h.hexdigest()[:24]}.jsonl")

    def load(self) -> dict[int, EngineResult]:
        if not self.file.exists():
            return {}
        done: dict[int, EngineResult] = {}
        for line in self.file.read_text("utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short by the crash itself; that chunk is redone
            done[row["index"]] = EngineResult(row["language"], [RawSegment(**s) for s in row["segments"]])
        if done:
            log.info("resuming %s: %d chunk(s) already done", self.file.name, len(done))
        return done

    def save(self, index: int, result: EngineResult) -> None:
        row = {"index": index, "language": result.language, "segments": [asdict(s) for s in result.segments]}
        with self.file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
