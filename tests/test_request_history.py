from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from test_model_service import FakeRuntime, call_http, config

from local_model_service.broker import RequestBroker
from local_model_service.history import RETENTION_SECONDS, RequestHistory
from local_model_service.server import ModelHTTPServer


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if payload.get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": "hi"}}]},
                {"choices": [], "timings": {"prompt_n": 7, "predicted_n": 3}},
            ]
            body = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
            body += b"data: [DONE]\n\n"
            content_type = "text/event-stream"
        else:
            body = json.dumps(
                {
                    "model": payload["model"],
                    "choices": [],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 5},
                }
            ).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_history_records_metadata_tokens_and_errors_without_prompts(tmp_path) -> None:
    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    runtime = FakeRuntime()
    runtime.port, runtime.token = backend.server_port, "test"
    broker = RequestBroker(config(tmp_path), runtime)
    history = RequestHistory(tmp_path / "requests.db")
    server = ModelHTTPServer(broker, history=history)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    message = [{"role": "user", "content": "secret prompt text"}]
    try:
        assert (
            call_http(
                server,
                "/v1/chat/completions",
                {"model": "test-model", "messages": message},
                {"User-Agent": "reader/1"},
            )[0]
            == 200
        )
        assert (
            call_http(
                server,
                "/v1/chat/completions",
                {"model": "test-model", "messages": message, "stream": True},
                {"X-Client-Name": "pi", "X-Request-Priority": "3"},
            )[0]
            == 200
        )
        assert (
            call_http(server, "/v1/chat/completions", {"model": "other", "messages": message})[0]
            == 404
        )
        status, body, _ = call_http(server, "/history?hours=1")
        assert status == 200
        summary = json.loads(body)
        assert summary["totals"] == {
            "requests": 3,
            "succeeded": 2,
            "failed": 1,
            "prompt_tokens": 18,
            "completion_tokens": 8,
            "average_ms": summary["totals"]["average_ms"],
            "average_success_ms": summary["totals"]["average_success_ms"],
        }
        failed, streamed, plain = summary["recent"]
        assert (failed["caller"], failed["status"], failed["error_code"]) == (
            "未标识",
            404,
            "MODEL_NOT_FOUND",
        )
        assert (streamed["caller"], streamed["stream"], streamed["priority"]) == ("pi", True, 3)
        assert (streamed["prompt_tokens"], streamed["completion_tokens"]) == (7, 3)
        assert (plain["caller"], plain["prompt_tokens"], plain["queue_ms"] is not None) == (
            "reader/1",
            11,
            True,
        )
        assert {c["caller"] for c in summary["callers"]} == {"pi", "reader/1", "未标识"}
        assert call_http(server, "/history?hours=0")[0] == 400
        assert call_http(server, "/history", headers={"Origin": "http://evil.test"})[0] == 403
    finally:
        server.shutdown()
        server.server_close()
        broker.close()
        backend.shutdown()
        backend.server_close()
        history.close()
    assert b"secret prompt text" not in (tmp_path / "requests.db").read_bytes()


def test_history_drops_records_past_retention(tmp_path) -> None:
    history = RequestHistory(tmp_path / "requests.db")
    now = time.time()
    for started in (now - RETENTION_SECONDS - 60, now - 60):
        history.record(
            started_at=started,
            caller="x",
            model="m",
            stream=False,
            priority=0,
            status=200,
            total_ms=5,
        )
    history.prune(now)
    assert history.summary(24 * 60, now=now)["totals"]["requests"] == 1
    history.close()
