"""The shapes that leave the pipeline. Times are seconds from the start of the file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    confidence: float | None = None  # 0..1, from the model's average token log-probability


@dataclass
class Transcript:
    language: str | None
    duration: float
    engine: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "duration": round(self.duration, 3),
            "engine": self.engine,
            "text": self.text,
            "segments": [asdict(s) for s in self.segments],
        }
