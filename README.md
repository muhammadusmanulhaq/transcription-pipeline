# Transcription pipeline

Audio in (any format, any length), text out with a timestamp on every segment.

## Design decisions

Choosing a model was the easy part. The work went into what breaks in practice:
- files in formats nobody planned for;
- recordings hours long;
- jobs that die halfway.

Each decision below gives its reason and what it costs.

1. **Parakeet TDT 0.6B v3, quantised to int8, run on ONNX Runtime.**
   - *Why:* on CPU it is several times faster than Whisper base and more accurate. A
     27 s clip took 6 s, against 34 s for Whisper. In int8 it is 640 MB instead of
     2.5 GB, and the ONNX export installs without PyTorch, CUDA or NeMo.
   - *Cost:* it covers 25 European languages, and it reports neither the language nor
     a confidence score.

2. **The pipeline depends on an `Engine` interface, not on a library.**
   - *Why:* Parakeet, Whisper (the fallback for its ~100 languages, Arabic included)
     and a mock engine are interchangeable. Changing model or moving to a hosted API
     means writing one class. The mock reports real timestamps, so the tests check
     the pipeline's own logic and need no model download.

3. **One decoder for every format, then normalise immediately.**
   - *Why:* PyAV bundles FFmpeg, so WAV, MP3, M4A, FLAC, Opus and MP4 audio share one
     code path, with no per-format branches and no system ffmpeg to install.
   - Everything becomes mono float32 at 16 kHz on the way in. Nothing downstream knows
     the original format, so nothing downstream can get it wrong.

4. **Validate before queueing.**
   - *Why:* every upload is probed first. A file that isn't media, or has no audio
     stream, gets `415` in the request that sent it. Otherwise it would fail minutes
     later inside a job nobody is watching.
   - The extension is ignored: the content decides.

5. **Trust decoded samples, not file headers.**
   - *Why:* durations and timestamps are counted from the decoded samples, because VBR
     MP3 headers are often wrong. A damaged packet is skipped and logged, so it costs
     milliseconds of audio rather than the whole transcript.

6. **Stream the audio; never load the whole file.**
   - *Why:* an hour of 16 kHz float32 is ~230 MB. Decoding as a stream keeps memory
     at one chunk whatever the length.

7. **Cut chunks at pauses, not into overlapping windows.**
   - *Why:* each cut lands at the quietest 100 ms in the last 15 s of a ~5-minute
     window. Chunks don't overlap, so no word is split and there is no fragile
     overlap-and-deduplicate step.
   - *Cost:* now and then a cut falls at a comma pause, and one sentence comes out as
     two segments, with every word intact. On an 11-minute test this happened to
     1 sentence in 125.

8. **Let the VAD decide segment boundaries.**
   - *Why:* Silero VAD splits each chunk at pauses of 500 ms or more (sentence
     boundaries, not breaths) and keeps every segment under 30 s, the range where
     Parakeet is most accurate. Silence never reaches the model, so it can't
     invent text there.

9. **Checkpoint every finished chunk.**
   - *Why:* a crash late in a long file resumes at the chunk it died on, not at zero.
   - The checkpoint key covers the file's content hash, the engine, the language and
     the chunk length, so it is never reused for a different input.

10. **Asynchronous jobs, not a blocking request.**
    - *Why:* long files take minutes, longer than any proxy will hold a connection open.
      `POST` returns `202` and a job id; `GET` reports progress and then the result.

11. **A fixed output contract.**
    - *Why:* consumers shouldn't each re-clean the output. Downstream code can rely on:
      - ids run 1..n;
      - segments are ordered and inside the file's duration;
      - text is trimmed and never empty;
      - a model looping on itself is dropped.
    - One result renders as JSON, SRT, VTT or plain text.

12. **Deliberately single-machine for now.**
    - Jobs are held in memory and one worker thread runs one model, which is enough to
      prove the pipeline. [System design at scale](#system-design-at-scale-not-implemented)
      describes the queue, object storage and database that replace them. The chunk
      checkpoints are already the piece that makes distributed recovery work.

```
file ─▶ probe ─▶ decode (stream) ─▶ chunk at pauses ─▶ Parakeet + VAD ─▶ re-time ─▶ clean ─▶ JSON / SRT / VTT / TXT
```

- **Engine:** NVIDIA **Parakeet TDT 0.6B v3**, int8 (~640 MB), run on ONNX Runtime through
  [`onnx-asr`](https://github.com/istupakov/onnx-asr). It needs no PyTorch or CUDA and runs on CPU.
  Silero VAD supplies the segment boundaries.
  Whisper (`faster-whisper`) is a second engine behind the same interface (`--engine whisper`).
  It covers ~100 languages, Arabic among them. Parakeet v3 covers 25 European languages.
- **Measured on a laptop CPU:** 11 minutes of speech in 170 s (3.9x faster than real time).
  Whisper base took 34 s for a 27 s clip that Parakeet did in 6 s.

## Run it

```bash
uv venv && uv pip install -e ".[dev]"

# CLI
python -m transcriber talk.mp3                 # JSON to stdout
python -m transcriber talk.m4a -f srt -o talk.srt
python -m transcriber lecture.mp3 --checkpoint-dir .ckpt   # resumable

# Service + upload page
python -m uvicorn transcriber.api:app --port 8000
# open http://localhost:8000: drop in audio or video, watch progress, click a line to hear it
curl -F file=@talk.mp3 localhost:8000/transcriptions        # -> 202 {"id": ...}
curl localhost:8000/transcriptions/<id>                     # status, progress, result
curl "localhost:8000/transcriptions/<id>?format=srt"

pytest        # 17 tests, mock engine, no model download
```

The first run downloads the model from Hugging Face. Environment for the service:
`TRANSCRIBER_ENGINE` (`parakeet` | `whisper` | `mock`), `TRANSCRIBER_MODEL`,
`TRANSCRIBER_CHUNK_SECONDS` (300), `TRANSCRIBER_MAX_UPLOAD_MB` (2048).

## Output

```json
{
  "language": null,
  "duration": 26.6,
  "engine": "parakeet-tdt-0.6b-v3/int8",
  "text": "Welcome to the quarterly planning meeting. Today we will ...",
  "segments": [
    {"id": 1, "start": 0.13, "end": 2.206, "text": "Welcome to the quarterly planning meeting.", "confidence": null},
    {"id": 2, "start": 2.978, "end": 5.534, "text": "Today we will review the transcription pipeline.", "confidence": null}
  ]
}
```

Times are seconds from the start of the file. Every consumer can rely on the following:
- ids run 1..n;
- segments are ordered and inside `[0, duration]`;
- text is trimmed and never empty;
- a segment repeating the previous one word for word has been dropped.

## How are different audio formats handled?

- **One decoder.** PyAV bundles FFmpeg, so WAV, MP3, M4A/AAC, FLAC, OGG/Opus, WebM and
  the audio track of an MP4 all take the same path. There is no per-format code and no
  system ffmpeg to install.
- **Normalise once.** Everything becomes mono float32 at 16 kHz straight away: stereo is
  down-mixed and any sample rate is resampled. Nothing after decoding knows what the
  input was.
- **Refuse early.** The upload is probed before it is queued. A file that isn't media,
  or has no audio stream, gets `415` with a reason, whatever its extension says.
- **Trust the samples, not the header.** Duration is counted from the decoded samples,
  because VBR MP3 headers are often wrong. A packet that won't decode is skipped and
  logged instead of failing the whole file.

## How are long files handled?

- **Stream, don't load.** An hour of 16 kHz float32 is ~230 MB. The file is decoded as a
  stream and cut into chunks (5 min by default), so memory stays at one chunk whatever
  the length.
- **Cut at pauses.** Each cut lands at the quietest 100 ms in the last 15 s of the window.
  Chunks are contiguous and don't overlap, so no word is split and no
  overlap-and-deduplicate step is needed.
- **Re-time.** Each chunk's segments are shifted by the chunk's start time and clamped
  inside the chunk, so the output is on the file's timeline. Inside a chunk, VAD keeps
  every segment to 30 s or less, the range where Parakeet is most accurate.
- **Resume.** With a checkpoint directory, each finished chunk is appended to a file keyed
  by the file's content hash plus the settings. A crash late in a long file resumes at
  the chunk it died on.
- **Asynchronous.** The API returns `202` and a job id at once; `GET` reports `progress`
  until the result is ready. No proxy holds a connection open for 20 minutes.
- **Whisper only:** the language detected on the first chunk is kept for the rest, and
  the tail of each chunk's text is passed to the next as a prompt, so spelling stays
  consistent across cuts. Parakeet has neither option.

Checked on the 11-minute file in 60 s chunks: 125 sentences, 124 transcribed exactly as
spoken. In the other one, a cut fell in the pause at a comma, so it came out as two
segments with every word intact.

## System design at scale (not implemented)

The code above runs on one machine: jobs in memory, one worker thread. This is how it
would grow.

```
client ──▶ API (stateless, N replicas) ──▶ Postgres (jobs, transcripts)
   │            │
   │            └──▶ queue (SQS / Redis) ──▶ GPU workers ──▶ object storage (audio, exports)
   └──── presigned PUT ──────────────────────────────────▶ object storage
```

### Concurrent uploads

- The audio goes straight from the client to object storage with a presigned URL.
  The API only creates a job row and enqueues it, so it stays stateless and scales
  out behind a load balancer. A 2 GB upload never passes through an API server.
- A queue sits between the API and a pool of GPU workers. A traffic spike becomes a
  longer queue, not an overloaded model. Workers pull work at their own pace, and the
  pool autoscales on queue depth.
- Each worker loads the model once and runs one job per GPU. Several chunks are
  batched through the model together.
- Fairness and backpressure: per-API-key rate limits and concurrent-job caps, and a
  `429` with `Retry-After` when the queue is too deep. One customer's 50-hour backlog
  can't starve everyone else.
- Long files are split into chunk tasks (the chunker already cuts at pauses), so one
  file can run on several workers in parallel and finish in the time of its slowest
  chunk.

### Storing audio and transcripts

- **Audio** goes in object storage (S3, R2, GCS), keyed by its SHA-256. Uploading the
  same file twice stores it once and can return the existing transcript. A lifecycle
  rule deletes raw audio after N days: it is large, sensitive, and not needed once
  transcribed.
- **Jobs** go in Postgres: status, progress, attempts, error, engine and model version,
  timestamps, owner.
- **Transcripts** also go in Postgres: the full JSON as JSONB, plus a `segments` table
  `(job_id, idx, start, end, text, confidence)` with a full-text index. Search and
  "jump to the moment X was said" are then plain SQL. SRT/VTT exports are rendered once
  and cached in object storage.
- **Checkpoints** (one row or object per finished chunk) live in the same store, so any
  worker can resume any job.
- Every row records the model and version that produced it, so transcripts can be
  re-run selectively when the model is upgraded. Data is encrypted at rest and access
  is scoped per tenant.

### Retrying and recovering failed transcriptions

- **Crashes are covered by the queue.** A worker acknowledges a message only after the
  result is saved. If the worker dies, the visibility timeout expires and another worker
  picks the job up.
- **Work isn't redone.** Jobs are idempotent and checkpointed per chunk (as already
  implemented), so the retry resumes at the last finished chunk.
- **Errors are classified.** Transient failures (GPU out of memory, network, storage
  timeouts) retry with exponential backoff and jitter, up to N attempts. Permanent ones
  (corrupt or non-audio file, unsupported language) fail at once with a clear reason;
  most are caught by the probe at upload anyway.
- **Nothing is lost silently.** A job that exhausts its retries goes to a dead-letter
  queue, is marked `failed` with the last error, and alerts. It can be replayed after a
  fix.
- **Hung jobs are caught.** A per-job timeout scales with audio duration, and a sweeper
  re-queues any job stuck in `running` past its heartbeat.

### Exposing it as an API

An asynchronous job API, because transcription takes longer than an HTTP request should
stay open.

- `POST /v1/transcriptions` takes a multipart upload, a `source_url`, or an `upload_id`
  from a presigned upload, plus options (`language`, `engine`, `webhook_url`). It
  answers `202 {id, status}`. An `Idempotency-Key` header makes client retries safe.
- `GET /v1/transcriptions/{id}` returns status, progress and the result.
  `?format=json|srt|vtt|txt` (already implemented).
- `GET /v1/transcriptions` lists your jobs; `DELETE /v1/transcriptions/{id}` removes the
  audio and the transcript.
- **Webhooks** are signed with HMAC and POSTed on completion or failure, so clients
  don't have to poll. Polling still works.
- **Optional streaming:** a WebSocket endpoint for live audio, which sends partial
  segments as speech is recognised.
- **Around the endpoints:** API-key or OAuth auth, versioning in the path, OpenAPI docs
  (generated by FastAPI), consistent error bodies, and per-key quotas.

### Other next steps
- word-level timestamps (`with_timestamps()` in onnx-asr);
- speaker diarization (pyannote);
- a confidence score for Parakeet, which the ONNX adapter doesn't expose yet.
