"""Test audio is synthesised: tone bursts at known times, in known formats.

The mock engine reports where it hears sound, so a test can check that a
burst placed at 412.0s in the file comes out at 412.0s in the transcript,
whatever the format, the sample rate, or the chunk it fell in.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest


def tone_track(bursts: list[tuple[float, float]], total: float, rate: int) -> np.ndarray:
    """Silence of `total` seconds with a 440 Hz tone over each (start, end)."""
    t = np.arange(int(total * rate)) / rate
    signal = np.zeros_like(t, dtype=np.float32)
    for start, end in bursts:
        mask = (t >= start) & (t < end)
        signal[mask] = 0.5 * np.sin(2 * np.pi * 440 * t[mask])
    return signal


def write_audio(path: Path, samples: np.ndarray, rate: int, codec: str, channels: int = 1) -> Path:
    layout = "stereo" if channels == 2 else "mono"
    with av.open(str(path), "w") as out:
        stream = out.add_stream(codec, rate=rate, layout=layout)
        data = np.tile(samples, (channels, 1)).astype(np.float32)  # planar: one row per channel
        step = 1024
        for i in range(0, data.shape[1], step):
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(data[:, i : i + step]), format="fltp", layout=layout)
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return path


@pytest.fixture
def make_file(tmp_path):
    def make(name: str, bursts, total: float, *, rate=16_000, codec="pcm_s16le", channels=1) -> Path:
        return write_audio(tmp_path / name, tone_track(bursts, total, rate), rate, codec, channels)

    return make
