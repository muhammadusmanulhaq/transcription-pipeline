import pytest

from transcriber.engines import EngineResult, MockEngine, RawSegment
from transcriber.models import Segment, Transcript
from transcriber.pipeline import transcribe_file
from transcriber.postprocess import clean, to_srt, to_vtt


def test_timestamps_are_on_the_files_timeline_across_chunks(make_file):
    # 11 minutes, 30 s chunks: bursts land in many different chunks.
    bursts = [(s, s + 1.5) for s in (2.0, 61.0, 125.0, 299.0, 412.0, 640.0)]
    path = make_file("long.mp3", bursts, total=660.0, rate=44_100, codec="libmp3lame", channels=2)

    t = transcribe_file(path, MockEngine(), chunk_seconds=30)

    assert len(t.segments) == len(bursts)
    for seg, (start, end) in zip(t.segments, bursts):
        assert seg.start == pytest.approx(start, abs=0.06)
        assert seg.end == pytest.approx(end, abs=0.06)
    assert [s.id for s in t.segments] == list(range(1, len(bursts) + 1))
    assert t.duration == pytest.approx(660.0, abs=0.1)


def test_language_found_in_the_first_chunk_is_kept(make_file):
    path = make_file("x.wav", [(1, 2), (40, 41)], total=60.0)
    seen = []

    class Recorder(MockEngine):
        def transcribe(self, audio, *, language=None, prompt=None):
            seen.append((language, prompt))
            return EngineResult("ar", [RawSegment(0.5, 1.0, "مرحبا")])

    t = transcribe_file(path, Recorder(), chunk_seconds=20)

    assert seen[0] == (None, None)
    assert all(lang == "ar" for lang, _ in seen[1:])  # pinned after detection
    assert all(prompt == "مرحبا" for _, prompt in seen[1:])  # previous text carried forward
    assert t.language == "ar"


def test_a_rerun_resumes_from_the_checkpoint(make_file, tmp_path):
    path = make_file("x.wav", [(1, 2), (25, 26), (45, 46)], total=55.0)  # three chunks of <= 20 s

    class Crashes(MockEngine):
        def transcribe(self, audio, **kw):
            if self.calls == 2:
                raise RuntimeError("worker killed")
            return super().transcribe(audio, **kw)

    with pytest.raises(RuntimeError):
        transcribe_file(path, Crashes(), chunk_seconds=20, checkpoint_dir=tmp_path / "ck")

    engine = MockEngine()
    t = transcribe_file(path, engine, chunk_seconds=20, checkpoint_dir=tmp_path / "ck")
    assert engine.calls == 1  # two of three chunks came from the checkpoint
    assert len(t.segments) == 3


def test_clean_removes_what_downstream_should_not_see():
    raw = [
        Segment(0, 5.0, 6.0, "  second   line "),
        Segment(0, 1.0, 2.0, "first"),
        Segment(0, 2.0, 2.5, "   "),
        Segment(0, 6.0, 7.0, "Second line"),  # Whisper looping on itself
        Segment(0, 9.0, 12.0, "past the end"),
    ]
    out = clean(raw, duration=10.0)
    assert [(s.id, s.start, s.end, s.text) for s in out] == [
        (1, 1.0, 2.0, "first"),
        (2, 5.0, 7.0, "second line"),
        (3, 9.0, 10.0, "past the end"),
    ]


def test_subtitle_formats():
    t = Transcript("en", 3725.5, "mock", [Segment(1, 3723.25, 3725.5, "hello")])
    assert to_srt(t) == "1\n01:02:03,250 --> 01:02:05,500\nhello\n"
    assert to_vtt(t) == "WEBVTT\n\n01:02:03.250 --> 01:02:05.500\nhello\n"
