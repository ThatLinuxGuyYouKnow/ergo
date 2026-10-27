"""Headless API surface: submit -> poll -> result + s3 links."""
import os
import time

from fastapi.testclient import TestClient

from ergo.api import app, _event_public, _result_view


def _client():
    return TestClient(app)


def test_openapi_has_result_endpoint():
    spec = _client().get("/openapi.json").json()
    assert "/api/jobs/{jid}/result" in spec["paths"]


def test_event_public_backfills_urls():
    e = _event_public({"clip": "jobs/x/event_1.mp4", "score": 0.9,
                       "decision": "STORE",
                       "result": {"uri": "s3://b/k/clip.mp4"}})
    assert e["clip_url"] == "/files/jobs/x/event_1.mp4"
    assert e["s3_uri"] == "s3://b/k/clip.mp4"


def test_result_view_shapes_s3_links():
    job = {"id": "abc", "name": "v.mp4", "status": "done", "events": [
        {"clip": "jobs/abc/e1.mp4", "decision": "STORE",
         "result": {"uri": "s3://b/k1"}},
        {"clip": "jobs/abc/e2.mp4", "decision": "DISCARD", "result": {}},
    ], "annotated": "jobs/abc/annotated.mp4"}
    v = _result_view(job)
    assert v["annotated_url"] == "/files/jobs/abc/annotated.mp4"
    assert v["s3_links"] == ["s3://b/k1"]
    assert v["n_events"] == 2


def test_submit_poll_result_headless():
    src = "data/env2/video_env_2_anonymized/jv-scrivania-terra-001.mp4"
    if not os.path.exists(src):
        return
    c = _client()
    with open(src, "rb") as f:
        r = c.post("/api/jobs", files={"file": ("t.mp4", f, "video/mp4")},
                   data={"threshold": "0.4"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert r.json()["result_url"] == f"/api/jobs/{jid}/result"
    for _ in range(120):
        s = c.get(f"/api/jobs/{jid}").json()
        if s.get("status") in ("done", "error"):
            break
        time.sleep(1)
    assert s["status"] == "done"
    res = c.get(f"/api/jobs/{jid}/result").json()
    assert res["job_id"] == jid
    assert res["annotated_url"] and res["annotated_url"].startswith("/files/")
    assert isinstance(res["events"], list) and res["n_events"] >= 1
    assert isinstance(res["s3_links"], list)
    for e in res["events"]:
        assert e["clip_url"].startswith("/files/")
