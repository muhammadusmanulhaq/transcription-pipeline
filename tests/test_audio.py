import numpy as np
import pytest

from transcriber import audio
from transcriber.audio import SAMPLE_RATE

BURSTS = [(1.0, 2.0), (3.5, 5.0)]


@pytest.mark.parametrize(
    "name, codec, rate, channels",
    [
        ("a.wav", "pcm_s16le", 16_000, 1),
        ("b.wav", "pcm_s16le", 44_100, 2),  # CD audio: resampled and down-mixed
        ("c.mp3", "libmp3lame", 44_100, 2),
        ("d.flac", "flac", 48_000, 1),
        ("e.ogg", "libopus", 48_000, 2),
        ("f.m4a", "aac", 22_050, 1),
    ],
)
def test_every_format_decodes_to_the_same_shape(make_file, name, codec, rate, channels):
    path = make_file(name, BURSTS, total=6.0, rate=rate, codec=codec, channels=channels)

    info = audio.probe(path)
    assert info.sample_rate == rate and info.channels == channels

    samples = np.concatenate(list(audio.decode(path)))
    assert samples.dtype == np.float32 and samples.ndim == 1
    # Lossy codecs pad a few ms at the edges; the length must still be the file's.
    assert abs(len(samples) / SAMPLE_RATE - 6.0) < 0.1


def test_a_text_file_is_refused(tmp_path):
    bogus = tmp_path / "notes.mp3"  # the extension lies; the content decides
    bogus.write_text("this is not audio")
    with pytest.raises(audio.UnsupportedAudio):
        audio.probe(bogus)


def test_chunks_are_contiguous_bounded_and_cut_in_silence():
    rate = SAMPLE_RATE
    # 10 minutes of "speech": loud everywhere except 1 s pauses every 40 s.
    t = np.arange(600 * rate) / rate
    signal = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    pauses = [(p, p + 1.0) for p in range(39, 600, 40)]
    for a, b in pauses:
        signal[int(a * rate) : int(b * rate)] = 0.0

    blocks = np.array_split(signal, 3000)  # arrives in small frames, like a decoder
    chunks = list(audio.chunk(blocks, target_seconds=120, search_seconds=15))

    assert sum(len(c.samples) for c in chunks) == len(signal)  # nothing lost or doubled
    for prev, nxt in zip(chunks, chunks[1:]):
        assert abs(prev.end - nxt.start) < 1e-9  # no gap, no overlap
    assert all(c.duration <= 120 for c in chunks)
    for c in chunks[:-1]:  # every cut but the end of the file falls inside a pause
        assert any(a <= c.end <= b for a, b in pauses), c.end
