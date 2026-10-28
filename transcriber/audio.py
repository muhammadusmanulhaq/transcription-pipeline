"""Reading audio: probe, decode, and cut into chunks.

Every format goes through one decoder. PyAV ships its own FFmpeg build, so
WAV, MP3, M4A/AAC, FLAC, OGG/Opus, WebM, and the audio track of an MP4 are
read the same way, with no system ffmpeg to install. Whatever the input,
the decoded output is the same thing: mono float32 PCM at 16 kHz, which is
what Whisper-family models expect.

Decoding is a stream, not a load. A two-hour file at 16 kHz float32 is about
460 MB in memory; read as a stream and cut into chunks, the pipeline never
holds more than one chunk at a time, however long the recording is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import av
import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


class UnsupportedAudio(ValueError):
    """The file can't be turned into audio: not media, or no audio stream."""


@dataclass(frozen=True)
class AudioInfo:
    container: str
    codec: str
    sample_rate: int
    channels: int
    duration: float | None  # seconds; None when the container doesn't record it


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float  # seconds from the start of the original file
    samples: np.ndarray  # mono float32 at SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE

    @property
    def end(self) -> float:
        return self.start + self.duration


def probe(path: str | Path) -> AudioInfo:
    """Check that `path` holds decodable audio and describe it.

    Runs before any work is queued, so a bad upload is refused at once
    instead of failing minutes later inside a job.
    """
    try:
        container = av.open(str(path))
    except av.error.FFmpegError as e:
        raise UnsupportedAudio(f"not a readable media file ({e.strerror or e})") from e

    with container:
        if not container.streams.audio:
            raise UnsupportedAudio("the file has no audio stream")
        stream = container.streams.audio[0]
        ctx = stream.codec_context

        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base

        return AudioInfo(
            container=container.format.name,
            codec=ctx.name,
            sample_rate=ctx.sample_rate,
            channels=ctx.channels,
            duration=duration,
        )


def decode(path: str | Path, sample_rate: int = SAMPLE_RATE) -> Iterator[np.ndarray]:
    """Yield the file's first audio track as mono float32 at `sample_rate`.

    Stereo is down-mixed and every source rate resampled by FFmpeg's
    resampler. A packet that fails to decode is skipped and counted rather
    than failing the file: a damaged frame should cost a few milliseconds of
    audio, not the whole transcript.
    """
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
    skipped = 0
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                skipped += 1
                continue
            for frame in frames:
                for out in resampler.resample(frame):
                    yield out.to_ndarray().reshape(-1)
        for out in resampler.resample(None):  # flush what the resampler buffered
            yield out.to_ndarray().reshape(-1)
    if skipped:
        log.warning("%s: skipped %d undecodable packet(s)", path, skipped)


def chunk(
    blocks: Iterable[np.ndarray],
    *,
    target_seconds: float = 300.0,
    search_seconds: float = 15.0,
    sample_rate: int = SAMPLE_RATE,
) -> Iterator[Chunk]:
    """Cut a stream of samples into chunks of at most `target_seconds`.

    Each cut lands at the quietest moment in the last `search_seconds` of
    the window, which in speech is almost always a pause between words. The
    chunks are therefore contiguous and non-overlapping, and no
    overlap-and-deduplicate step is needed to stitch them back together.
    """
    max_len = int(target_seconds * sample_rate)
    search_len = min(int(search_seconds * sample_rate), max_len // 2)

    pending: list[np.ndarray] = []
    pending_len = 0
    emitted = 0  # samples already handed out, i.e. where the next chunk starts
    index = 0

    for block in blocks:
        pending.append(block)
        pending_len += len(block)
        while pending_len >= max_len:
            buf = np.concatenate(pending)
            cut = _quietest_point(buf, max_len - search_len, max_len, sample_rate)
            yield Chunk(index, emitted / sample_rate, buf[:cut])
            index += 1
            emitted += cut
            pending, pending_len = [buf[cut:]], len(buf) - cut

    if pending_len:
        yield Chunk(index, emitted / sample_rate, np.concatenate(pending))


def _quietest_point(samples: np.ndarray, lo: int, hi: int, sample_rate: int) -> int:
    """The middle of the lowest-energy 100 ms frame in samples[lo:hi].

    Ties go to the latest frame, so a window that is silent throughout still
    yields a chunk of nearly the full target length.
    """
    frame = sample_rate // 10
    window = samples[lo:hi]
    n = len(window) // frame
    if n == 0:
        return hi
    energy = np.square(window[: n * frame].reshape(n, frame)).mean(axis=1)
    latest = n - 1 - int(np.argmin(energy[::-1]))
    return lo + latest * frame + frame // 2
