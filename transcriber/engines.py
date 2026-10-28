"""Speech-to-text engines.

The pipeline talks to an `Engine`, not to a library. Parakeet does the real
work, Whisper is the alternative (it covers far more languages, Arabic
among them), and `MockEngine` stands in for tests and for running the
service without downloading a model. Moving to a hosted API or a different model means
writing one more class with the same `transcribe` method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .audio import SAMPLE_RATE


@dataclass
class RawSegment:
    start: float  # seconds from the start of the chunk, not of the file
    end: float
    text: str
    confidence: float | None = None


@dataclass
class EngineResult:
    language: str | None
    segments: list[RawSegment] = field(default_factory=list)


class Engine(Protocol):
    name: str

    def transcribe(
        self, audio: np.ndarray, *, language: str | None = None, prompt: str | None = None
    ) -> EngineResult: ...


class ParakeetEngine:
    """NVIDIA Parakeet TDT 0.6B, run through onnx-asr on ONNX Runtime.

    The default engine. On CPU it runs several times faster than Whisper
    base and is more accurate, and int8 weights (~640 MB) keep it within
    laptop memory. Using the ONNX export rather than NeMo means no PyTorch
    or CUDA toolchain, so it installs anywhere.

    Silero VAD cuts each chunk into speech segments before recognition.
    Those segments are the timestamps, and silence never reaches the model.
    v3 covers 25 European languages and finds the language itself, but does
    not report it (so `language` stays None unless the caller sets one) and
    takes no prompt.
    """

    def __init__(self, model: str = "nemo-parakeet-tdt-0.6b-v3", quantization: str | None = "int8"):
        import onnx_asr  # heavy import, only when it's actually used

        self.name = f"{model.removeprefix('nemo-')}/{quantization or 'fp32'}"
        self._model = onnx_asr.load_model(model, quantization=quantization).with_vad(
            onnx_asr.load_vad("silero"),
            min_silence_duration_ms=500,  # split at sentence pauses, not at every breath
            max_speech_duration_s=30,  # Parakeet is most accurate on segments of 30 s or less
        )

    def transcribe(self, audio, *, language=None, prompt=None) -> EngineResult:
        segments = self._model.recognize(audio, sample_rate=SAMPLE_RATE)
        return EngineResult(
            language=language,
            segments=[RawSegment(s.start, s.end, s.text) for s in segments],
        )


class WhisperEngine:
    """faster-whisper: Whisper weights on CTranslate2, about 4x faster than the
    reference implementation on CPU, with int8 quantisation and a built-in
    Silero VAD.
    """

    def __init__(self, model: str = "base", device: str = "auto", compute_type: str = "default"):
        from faster_whisper import WhisperModel  # heavy import, only when it's actually used

        self.name = f"faster-whisper/{model}"
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio, *, language=None, prompt=None) -> EngineResult:
        segments, info = self._model.transcribe(
            audio,
            language=language,
            initial_prompt=prompt,
            beam_size=5,
            # Skips silence and music before decoding. That is faster, and it
            # stops Whisper inventing text ("Thanks for watching!") in quiet stretches.
            vad_filter=True,
        )
        return EngineResult(
            language=info.language,
            segments=[
                RawSegment(s.start, s.end, s.text, round(math.exp(s.avg_logprob), 3))
                for s in segments  # a generator: decoding happens as it is consumed
            ],
        )


class MockEngine:
    """Returns one segment per stretch of sound, labelled with where it is.

    Not speech recognition, but the timestamps are real, which is the part
    the pipeline is responsible for. The tests rely on that.
    """

    name = "mock"

    def __init__(self, threshold: float = 0.01, min_gap: float = 0.3):
        self.threshold = threshold
        self.min_gap = min_gap
        self.calls = 0

    def transcribe(self, audio, *, language=None, prompt=None) -> EngineResult:
        self.calls += 1
        frame = SAMPLE_RATE // 100  # 10 ms
        n = len(audio) // frame
        if n == 0:
            return EngineResult(language or "en")
        rms = np.sqrt(np.square(audio[: n * frame].reshape(n, frame)).mean(axis=1))
        loud = rms > self.threshold

        segments: list[RawSegment] = []
        i = 0
        while i < n:
            if not loud[i]:
                i += 1
                continue
            j = i
            while j < n and loud[j]:
                j += 1
            start, end = i * 0.01, j * 0.01
            if segments and start - segments[-1].end < self.min_gap:
                segments[-1].end = end
            else:
                segments.append(RawSegment(start, end, "", 1.0))
            i = j
        for k, s in enumerate(segments, 1):
            s.text = f"sound {k} ({s.start:.2f}-{s.end:.2f}s into the chunk)"
        return EngineResult(language or "en", segments)


def make_engine(kind: str = "parakeet", model: str | None = None) -> Engine:
    if kind == "parakeet":
        return ParakeetEngine(model or "nemo-parakeet-tdt-0.6b-v3")
    if kind == "whisper":
        return WhisperEngine(model or "base")
    if kind == "mock":
        return MockEngine()
    raise ValueError(f"unknown engine {kind!r} (use 'parakeet', 'whisper' or 'mock')")
