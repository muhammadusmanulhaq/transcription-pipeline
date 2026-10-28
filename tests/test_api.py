import time

import pytest
from fastapi.testclient import TestClient

from transcriber import api
from transcriber.engines import MockEngine


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "WORK_DIR", tmp_path)
    monkeypatch.setattr(api, "_engine", MockEngine())
    api.jobs.clear()
    return TestClient(api.app)


def wait(client, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/transcriptions/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_upload_poll_and_export(client, make_file):
    path = make_file("talk.mp3", [(1, 2)], total=4.0, rate=44_100, codec="libmp3lame")
    with path.open("rb") as f:
        r = client.post("/transcriptions", files={"file": ("talk.mp3", f, "audio/mpeg")})
    assert r.status_code == 202
    job = r.json()
    assert job["audio"]["codec"].startswith("mp3")  # FFmpeg's decoder is "mp3float"

    body = wait(client, job["id"])
    assert body["status"] == "done" and body["progress"] == 1.0
    assert body["result"]["segments"][0]["start"] == pytest.approx(1.0, abs=0.06)

    srt = client.get(f"/transcriptions/{job['id']}?format=srt")
    assert srt.headers["content-type"].startswith("application/x-subrip")
    assert "-->" in srt.text


def test_non_audio_is_refused_before_queueing(client):
    r = client.post("/transcriptions", files={"file": ("x.wav", b"hello", "audio/wav")})
    assert r.status_code == 415
    assert api.jobs == {}


def test_oversized_upload_is_refused(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_MB", 1)
    r = client.post("/transcriptions", files={"file": ("big.wav", b"\0" * (2 << 20), "audio/wav")})
    assert r.status_code == 413


def test_unknown_job(client):
    assert client.get("/transcriptions/nope").status_code == 404
