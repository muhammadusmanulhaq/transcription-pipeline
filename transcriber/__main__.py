"""Command line: python -m transcriber INPUT [-f json|srt|vtt|txt] [-o OUT]"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .audio import UnsupportedAudio, probe
from .engines import make_engine
from .pipeline import transcribe_file
from .postprocess import FORMATS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="transcriber", description="Transcribe an audio file with per-segment timestamps.")
    p.add_argument("input", type=Path)
    p.add_argument("-f", "--format", choices=sorted(FORMATS), default="json")
    p.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    p.add_argument("--engine", choices=["parakeet", "whisper", "mock"], default="parakeet")
    p.add_argument("--model", help="parakeet: nemo-parakeet-tdt-0.6b-v3 (default) or -v2; whisper: tiny, base (default), small, ...")
    p.add_argument("--language", help="ISO code such as en or ar; detected from the first chunk if omitted")
    p.add_argument("--chunk-seconds", type=float, default=300.0)
    p.add_argument("--checkpoint-dir", type=Path, help="save each finished chunk here so a rerun resumes")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    try:
        info = probe(args.input)
    except (UnsupportedAudio, FileNotFoundError) as e:
        print(f"error: {args.input}: {e}", file=sys.stderr)
        return 2
    print(
        f"{args.input.name}: {info.codec} in {info.container}, {info.sample_rate} Hz, "
        f"{info.channels} ch, {info.duration or 0:.1f}s",
        file=sys.stderr,
    )

    engine = make_engine(args.engine, args.model)
    began = time.perf_counter()  # after the model loads: this times transcription only
    transcript = transcribe_file(
        args.input,
        engine,
        language=args.language,
        chunk_seconds=args.chunk_seconds,
        checkpoint_dir=args.checkpoint_dir,
        on_progress=lambda f: print(f"\r{f:6.1%}", end="", file=sys.stderr, flush=True),
    )
    took = time.perf_counter() - began
    print(
        f"\n{len(transcript.segments)} segments, language {transcript.language}, "
        f"{took:.1f}s for {transcript.duration:.1f}s of audio ({transcript.duration / max(took, 1e-9):.1f}x real time)",
        file=sys.stderr,
    )

    rendered = FORMATS[args.format](transcript)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
