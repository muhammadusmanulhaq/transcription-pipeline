"""HTTP service.

    GET  /                           upload page: drop a file, watch progress, play it back by segment
    POST /transcriptions             upload a file -> 202 with a job id
    GET  /transcriptions/{id}        status, progress, and the transcript once done
    GET  /transcriptions/{id}?format=srt|vtt|txt
    GET  /health
    GET  /docs                       interactive API docs

Transcription is a job, not a request. A one-hour file takes minutes even on
a GPU, longer than any proxy will hold a connection open, so the upload
returns straight away and the client polls. The upload is checked (size,
decodable, has audio) before it is queued, so a bad file fails in the
request that sent it, not later inside a job.

The job table is in memory and there is one worker thread. That is enough
for a single machine. See README for what changes at scale.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from . import audio
from .engines import Engine, make_engine
from .models import Transcript
from .pipeline import transcribe_file
from .postprocess import FORMATS, MEDIA_TYPES

log = logging.getLogger(__name__)

ENGINE = os.environ.get("TRANSCRIBER_ENGINE", "parakeet")
MODEL = os.environ.get("TRANSCRIBER_MODEL") or None  # the engine's own default when unset
MAX_UPLOAD_MB = int(os.environ.get("TRANSCRIBER_MAX_UPLOAD_MB", "2048"))
CHUNK_SECONDS = float(os.environ.get("TRANSCRIBER_CHUNK_SECONDS", "300"))
WORK_DIR = Path(os.environ.get("TRANSCRIBER_WORK_DIR", Path(tempfile.gettempdir()) / "transcriber"))


@dataclass
class Job:
    id: str
    filename: str
    audio: dict
    status: str = "queued"  # queued -> running -> done | failed
    progress: float = 0.0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: Transcript | None = None

    def view(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if k != "result"}
        out["progress"] = round(self.progress, 3)
        if self.result is not None:
            out["result"] = self.result.to_dict()
        return out


app = FastAPI(title="Transcription service")
jobs: dict[str, Job] = {}
worker = ThreadPoolExecutor(max_workers=1)  # one model, one job at a time; a queue in front of it
_engine: Engine | None = None
_engine_lock = threading.Lock()


def engine() -> Engine:
    """Load the model once, on first use, and share it between jobs."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = make_engine(ENGINE, MODEL)
        return _engine


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "engine": ENGINE, "model": MODEL, "queued": sum(j.status == "queued" for j in jobs.values())}


@app.post("/transcriptions", status_code=202)
def create(file: UploadFile = File(...), language: str | None = Form(None)):
    job_id = uuid.uuid4().hex
    (WORK_DIR / "uploads").mkdir(parents=True, exist_ok=True)
    dest = WORK_DIR / "uploads" / (job_id + Path(file.filename or "").suffix.lower())

    # Streamed to disk in 1 MB pieces with the cap checked as it goes, so a
    # 2 GB upload never sits in memory and an oversized one stops early.
    limit = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with dest.open("wb") as out:
        while block := file.file.read(1 << 20):
            written += len(block)
            if written > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file is larger than {MAX_UPLOAD_MB} MB")
            out.write(block)

    try:
        info = audio.probe(dest)
    except audio.UnsupportedAudio as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(415, f"unsupported audio: {e}") from None

    job = Job(job_id, file.filename or dest.name, asdict(info))
    jobs[job_id] = job
    worker.submit(_run, job, dest, language or None)
    return {"id": job_id, "status": job.status, "audio": job.audio, "url": f"/transcriptions/{job_id}"}


@app.get("/transcriptions/{job_id}")
def read(job_id: str, format: str = Query("json", pattern="^(json|srt|vtt|txt)$")):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such transcription")
    if format == "json":
        return job.view()
    if job.status != "done":
        raise HTTPException(409, f"transcription is {job.status}; {format} is available once it is done")
    return PlainTextResponse(FORMATS[format](job.result), media_type=MEDIA_TYPES[format])


def _run(job: Job, path: Path, language: str | None) -> None:
    job.status = "running"
    try:
        job.result = transcribe_file(
            path,
            engine(),
            language=language,
            chunk_seconds=CHUNK_SECONDS,
            checkpoint_dir=WORK_DIR / "checkpoints",
            on_progress=lambda p: setattr(job, "progress", p),
        )
        job.status = "done"
    except Exception as e:  # a job's failure is reported on the job, never lost in a thread
        log.exception("job %s failed", job.id)
        job.status, job.error = "failed", f"{type(e).__name__}: {e}"
    finally:
        job.finished_at = time.time()
        path.unlink(missing_ok=True)
