from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_model_service.broker import RequestBroker, ServiceError
from local_model_service.config import ModelSettings, ServiceSettings
from local_model_service.runtime import LocalModelServerError, ManagedRuntime
from local_model_service.server import ModelHTTPServer


class FakeRuntime:
    def __init__(self) -> None:
        self.process = None
        self.identity = "a" * 64
        self.effective_alias = "test-model"
        self.loads = 0
        self.stops = 0

    def ensure_ready(self, cancel, deadline) -> None:
        self.loads += 1
        self.process = SimpleNamespace(pid=1)

    def stop(self) -> None:
        if self.process:
            self.stops += 1
        self.process = None

    def mark_failed(self, reason) -> None:
        self.stop()

    def describe(self) -> dict:
        return {
            "state": "ready" if self.process else "sleeping",
            "identity": self.identity,
            "effective_model": self.effective_alias,
            "requested_model": "test-model",
            "pid": 1 if self.process else None,
            "descriptor": {},
        }


def config(tmp_path: Path, **options) -> ServiceSettings:
    return ServiceSettings(
        port=0, assets_dir=str(tmp_path), model=ModelSettings(model="test-model"), **options
    )


def wait_for(predicate, timeout=2) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition timed out")
        time.sleep(0.01)


def call_http(server, path, payload=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request(
            "GET" if payload is None else "POST",
            path,
            body=json.dumps(payload).encode() if payload is not None else None,
            headers=headers or {},
        )
        response = connection.getresponse()
        return response.status, response.read(), response.headers
    finally:
        connection.close()


def test_queue_is_serial_bounded_priority_aware_and_drops_cancelled_requests(tmp_path) -> None:
    runtime = FakeRuntime()
    broker = RequestBroker(config(tmp_path, queue_capacity=2), runtime)
    entered = threading.Event()
    release = threading.Event()
    executed = []

    def execute(job):
        executed.append(job.payload["tag"])
        if job.payload["tag"] == "active":
            entered.set()
            release.wait(2)

    broker._execute = execute
    try:
        first = broker.submit({"model": "test-model", "tag": "active"})
        assert entered.wait(2)
        cancelled = broker.submit({"model": "test-model", "tag": "cancelled"})
        urgent = broker.submit({"model": "test-model", "tag": "urgent"}, priority=10)
        with pytest.raises(ServiceError, match="队列已满") as error:
            broker.submit({"model": "test-model", "tag": "overflow"})
        assert error.value.status == 429
        cancelled.cancel()
        release.set()
        assert first.done.wait(2) and urgent.done.wait(2) and cancelled.done.wait(2)
        assert executed == ["active", "urgent"]
        assert cancelled.error.code == "REQUEST_EXPIRED"
    finally:
        release.set()
        broker.close()


def test_metadata_queries_never_load_or_keep_an_idle_model_resident(
    tmp_path,
) -> None:
    runtime = FakeRuntime()
    broker = RequestBroker(config(tmp_path, idle_seconds=0.08), runtime)
    server = ModelHTTPServer(broker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def execute(job):
        runtime.ensure_ready(None, 0)
        job.result = {"model": "test-model"}

    broker._execute = execute
    try:
        for path in ("/health", "/status", "/v1/models"):
            assert call_http(server, path)[0] == 200
        assert runtime.loads == 0
        job = broker.submit({"model": "test-model"})
        assert job.done.wait(2)
        wait_for(lambda: runtime.stops == 1)
        assert runtime.process is None
        assert call_http(server, "/health")[0] == 200
        assert runtime.loads == 1
    finally:
        server.shutdown()
        server.server_close()
        broker.close()
        thread.join(2)


def test_queued_deadline_expires_without_loading_model(tmp_path) -> None:
    runtime = FakeRuntime()
    broker = RequestBroker(config(tmp_path, request_timeout_seconds=0.05), runtime)
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def execute(job):
        calls.append(job.sequence)
        entered.set()
        release.wait(2)

    broker._execute = execute
    try:
        broker.submit({"model": "test-model"})
        assert entered.wait(2)
        expired = broker.submit({"model": "test-model"})
        expired.deadline = time.monotonic() - 1
        release.set()
        assert expired.done.wait(2)
        assert expired.error.status == 504
        assert len(calls) == 1 and runtime.loads == 0
    finally:
        release.set()
        broker.close()


def test_identity_change_is_rejected_before_inference(tmp_path) -> None:
    runtime = FakeRuntime()
    broker = RequestBroker(config(tmp_path), runtime)
    try:
        job = broker.submit({"model": "test-model"}, expected_identity="b" * 64)
        assert job.done.wait(2)
        assert job.error.status == 409
        assert job.error.code == "MODEL_IDENTITY_CHANGED"
        assert job.connection is None
        with pytest.raises(ServiceError) as error:
            broker.submit({"model": "wrong-model"})
        assert error.value.code == "MODEL_NOT_FOUND"
    finally:
        broker.close()


def make_runtime(tmp_path: Path, *, draft=False) -> ManagedRuntime:
    model = tmp_path / "test.gguf"
    model.write_bytes(b"model")
    executable = tmp_path / "llama-server.exe"
    executable.write_bytes(b"executable")
    settings = ModelSettings(
        model="test-model",
        model_preset="custom",
        model_path=str(model),
        server_executable=str(executable),
        server_speculative_type="none",
        startup_timeout_seconds=1,
    )
    if draft:
        draft_path = tmp_path / "draft.gguf"
        draft_path.write_bytes(b"draft")
        settings.draft_model_path = str(draft_path)
        settings.server_speculative_type = "draft-dflash"
    return ManagedRuntime(
        ServiceSettings(assets_dir=str(tmp_path), model=settings, recovery_backoff_seconds=0.01),
        tmp_path / "service",
    )


def test_runtime_recovers_failed_launch_with_a_bound_and_preserves_failure_backoff(
    tmp_path,
) -> None:
    runtime = make_runtime(tmp_path)
    attempts = []

    def fail(cancel, deadline):
        attempts.append(1)
        raise LocalModelServerError("failed to load")

    runtime._launch = fail
    with pytest.raises(LocalModelServerError):
        runtime.ensure_ready(threading.Event(), time.monotonic() + 2)
    assert len(attempts) == 2
    assert runtime.state == "failed"
    with pytest.raises(LocalModelServerError, match="退避"):
        runtime.ensure_ready(threading.Event(), time.monotonic() + 2)
    assert len(attempts) == 2


def test_fallback_changes_identity_and_releases_failed_profile(tmp_path) -> None:
    runtime = make_runtime(tmp_path, draft=True)
    original = runtime.identity
    profiles = []

    def launch(cancel, deadline):
        profiles.append(runtime.effective_alias)
        if not runtime.fallback:
            raise LocalModelServerError("draft unsupported")
        runtime.state = "ready"

    runtime._launch = launch
    runtime.ensure_ready(threading.Event(), time.monotonic() + 2)
    assert profiles == ["test-model", "test-model-plain"]
    assert runtime.fallback and runtime.identity != original
    assert runtime.describe()["descriptor"]["draft_file"] is None


def test_weight_and_runtime_changes_get_distinct_identity(tmp_path) -> None:
    runtime = make_runtime(tmp_path)
    original = runtime.identity
    runtime.model_path.write_bytes(b"different model")
    runtime._refresh_identity()
    second = runtime.identity
    runtime.executable.write_bytes(b"different runtime")
    runtime._refresh_identity()
    assert len({original, second, runtime.identity}) == 3


def test_http_rejects_web_origins_nonlocal_host_and_unauthorized_control(tmp_path) -> None:
    runtime = FakeRuntime()
    broker = RequestBroker(config(tmp_path), runtime)
    server = ModelHTTPServer(broker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert call_http(server, "/health", headers={"Origin": "https://example.com"})[0] == 403
        assert call_http(server, "/health", headers={"Host": "attacker.example"})[0] == 403
        assert call_http(server, "/shutdown", {})[0] == 403
        assert not server.stop_event.is_set()
        assert call_http(server, "/v1/chat/completions", {"model": [], "messages": []})[0] == 400
        assert runtime.loads == 0
    finally:
        server.shutdown()
        server.server_close()
        broker.close()
        thread.join(2)


def test_config_disallows_public_binding_and_nonpositive_resource_limits(tmp_path) -> None:
    for change in ({"host": "0.0.0.0"}, {"idle_seconds": 0}, {"queue_capacity": 0}):
        candidate = config(tmp_path, **change)
        candidate.port = 8080
        with pytest.raises(ValueError):
            candidate.validate()


@pytest.fixture
def dashboard_gateway(tmp_path):
    runtime = FakeRuntime()
    broker = RequestBroker(config(tmp_path), runtime)
    server = ModelHTTPServer(broker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, broker, runtime
    finally:
        server.shutdown()
        server.server_close()
        broker.close()
        thread.join(2)


def dashboard_headers(server) -> dict:
    return {
        "Origin": f"http://127.0.0.1:{server.server_port}",
        "X-Service-Control-Token": server.dashboard_token,
        "X-Expected-Service-Instance": server.instance_id,
    }


def test_dashboard_is_local_read_only_without_a_private_ticket_and_never_loads(
    dashboard_gateway,
) -> None:
    server, _, runtime = dashboard_gateway
    status, body, headers = call_http(server, "/dashboard")
    assert status == 200
    assert "本机模型服务" in body.decode("utf-8")
    assert server.dashboard_token.encode() not in body
    assert server.control_token.encode() not in body
    assert headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert call_http(server, "/admin/status")[0] == 403
    assert call_http(server, "/admin/stop", {})[0] == 403
    assert not server.stop_event.is_set()
    assert runtime.loads == 0


@pytest.mark.parametrize("invalid", ["origin", "ticket", "instance", "host"])
def test_dashboard_rejects_cross_origin_stale_or_unauthenticated_controls(
    dashboard_gateway, invalid
) -> None:
    server, _, runtime = dashboard_gateway
    headers = dashboard_headers(server)
    if invalid == "origin":
        headers["Origin"] = "https://untrusted.example"
    elif invalid == "ticket":
        headers["X-Service-Control-Token"] = "wrong"
    elif invalid == "instance":
        headers["X-Expected-Service-Instance"] = "old-instance"
    else:
        headers["Host"] = f"untrusted.example:{server.server_port}"
    assert call_http(server, "/admin/stop", {}, headers=headers)[0] in {403, 409}
    assert not server.stop_event.is_set()
    assert runtime.loads == 0


def test_dashboard_busy_guard_does_not_interrupt_requests_and_freezes_restart_admission(
    dashboard_gateway,
) -> None:
    server, broker, runtime = dashboard_gateway
    entered, release = threading.Event(), threading.Event()
    restarted = []
    server.restart_handler = lambda: restarted.append(True)

    def execute(job):
        runtime.ensure_ready(None, 0)
        entered.set()
        release.wait(2)
        job.result = {"model": "test-model"}

    broker._execute = execute
    try:
        job = broker.submit({"model": "test-model"})
        assert entered.wait(2)
        for action in ("unload", "stop", "restart"):
            assert (
                call_http(server, f"/admin/{action}", {}, headers=dashboard_headers(server))[0]
                == 409
            )
        assert not restarted and not server.stop_event.is_set() and runtime.stops == 0
        release.set()
        assert job.done.wait(2)
        wait_for(lambda: not broker.status()["active"])
        assert call_http(server, "/admin/unload", {}, headers=dashboard_headers(server))[0] == 200
        assert runtime.process is None
        assert call_http(server, "/admin/restart", {}, headers=dashboard_headers(server))[0] == 200
        assert restarted == [True]
        assert not broker.status()["accepting_requests"]
        with pytest.raises(ServiceError, match="服务正在停止"):
            broker.submit({"model": "test-model"})
    finally:
        release.set()


def test_dashboard_restart_launch_failure_restores_admission(dashboard_gateway) -> None:
    server, broker, _ = dashboard_gateway

    def fail():
        raise OSError("could not launch helper")

    server.restart_handler = fail
    assert call_http(server, "/admin/restart", {}, headers=dashboard_headers(server))[0] == 503
    assert broker.status()["accepting_requests"]
    assert server.service_action == "" and not server.stop_event.is_set()


def test_dashboard_ticket_survives_restart_and_never_appears_in_cli_output(
    tmp_path, monkeypatch, capsys
) -> None:
    from local_model_service import cli

    token = cli.dashboard_token(tmp_path)
    assert cli.dashboard_token(tmp_path) == token
    events = []
    monkeypatch.setattr(cli, "start", lambda *_: events.append("started"))
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: events.append(url) or True)
    cli.open_dashboard(config(tmp_path), tmp_path / "service.json")
    assert events[0] == "started"
    assert events[1].endswith(f"/dashboard#{token}")
    assert token not in capsys.readouterr().out


@pytest.mark.parametrize("layout", ["beside-executable", "parent-directory"])
def test_packaged_service_uses_its_relocated_configuration(tmp_path, monkeypatch, layout) -> None:
    from local_model_service import config as service_config

    home = tmp_path / "old-home"
    (home / ".local-model-service").mkdir(parents=True)
    (home / ".local-model-service" / "service.json").write_text("{}")
    relocated = tmp_path / "independent service"
    executable_dir = relocated if layout == "beside-executable" else relocated / "app"
    executable_dir.mkdir(parents=True)
    (relocated / "service.json").write_text("{}")
    monkeypatch.delenv("LOCAL_MODEL_SERVICE_DIR", raising=False)
    monkeypatch.setattr(service_config.sys, "frozen", True, raising=False)
    monkeypatch.setattr(service_config.sys, "executable", str(executable_dir / "service.exe"))
    monkeypatch.setattr(Path, "home", lambda: home)
    assert service_config.default_service_dir() == relocated
    monkeypatch.setenv("LOCAL_MODEL_SERVICE_DIR", str(home / ".local-model-service"))
    assert service_config.default_service_dir() == home / ".local-model-service"


def test_relocated_dashboard_reports_its_active_directory_and_safe_launch_command(
    tmp_path, monkeypatch, dashboard_gateway
) -> None:
    from local_model_service import cli

    server, _, runtime = dashboard_gateway
    directory = tmp_path / "user's model service"
    config_path = directory / "service.json"
    monkeypatch.setattr(cli.sys, "frozen", True, raising=False)
    monkeypatch.setattr(cli.sys, "executable", str(directory / "app" / "service.exe"))
    server.config_path = config_path
    server.dashboard_command = cli._dashboard_command(config_path)
    status, body, _ = call_http(server, "/status")
    metadata = json.loads(body)
    assert status == 200
    assert metadata["service_directory"] == str(directory)
    assert "user''s model service" in metadata["dashboard_command"]
    assert "'--config'" in metadata["dashboard_command"]
    assert server.dashboard_token.encode() not in body
    assert runtime.loads == 0


def test_cli_stop_waits_for_listener_shutdown_and_is_idempotent(tmp_path) -> None:
    from local_model_service.cli import stop

    runtime = FakeRuntime()
    settings = config(tmp_path)
    broker = RequestBroker(settings, runtime)
    server = ModelHTTPServer(broker)
    settings.port = server.server_port
    (tmp_path / "control.json").write_text(
        json.dumps({"instance_id": server.instance_id, "token": server.control_token}),
        encoding="utf-8",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def close_on_stop():
        server.stop_event.wait(3)
        server.shutdown()
        server.server_close()

    closer = threading.Thread(target=close_on_stop, daemon=True)
    closer.start()
    try:
        stop(settings, tmp_path / "service.json")
        assert server.stop_event.is_set()
        stop(settings, tmp_path / "service.json")
    finally:
        server.stop_event.set()
        closer.join(3)
        broker.close()
        thread.join(2)


def test_backend_model_mismatch_never_admits_inference(tmp_path) -> None:
    runtime = make_runtime(tmp_path)
    runtime._backend_json = lambda *_: {"data": [{"id": "unexpected-model"}]}
    with pytest.raises(LocalModelServerError, match="身份不一致"):
        runtime._verify_loaded_model()


def test_bootstrap_preserves_assistant_data_and_exports_independent_settings(tmp_path) -> None:
    from local_model_service.cli import initialize

    runtime = make_runtime(tmp_path)
    assistant = tmp_path / "assistant.json"
    original = {
        "local_analysis": {
            "model": "test-model",
            "model_preset": "custom",
            "manage_server": True,
            "server_executable": str(runtime.executable),
            "model_path": str(runtime.model_path),
            "server_speculative_type": "none",
        },
        "source": {"title": "My book"},
    }
    assistant.write_text(json.dumps(original), encoding="utf-8")
    service_path = tmp_path / "service" / "service.json"
    initialized = initialize(service_path, assistant, tmp_path, connect_assistant=True)
    exported = ServiceSettings.load(service_path)
    changed = json.loads(assistant.read_text(encoding="utf-8"))
    assert exported.model.model_path == str(runtime.model_path)
    assert exported.model.server_speculative_type == "none"
    assert changed["source"] == original["source"]
    assert changed["local_analysis"]["manage_server"] is False
    assert changed["local_analysis"]["timeout_seconds"] > initialized.request_timeout_seconds
    backup = next(tmp_path.glob("assistant.json.*.bak"))
    assert json.loads(backup.read_text(encoding="utf-8")) == original


def test_force_bootstrap_refuses_to_overwrite_a_running_service(tmp_path, monkeypatch) -> None:
    from local_model_service.cli import initialize

    runtime = make_runtime(tmp_path)
    path = tmp_path / "service.json"
    runtime.config.save(path)
    original = path.read_bytes()
    monkeypatch.setattr(
        "local_model_service.cli._http", lambda *_: {"service": "local-model-service"}
    )
    with pytest.raises(ValueError, match="先停止"):
        initialize(path, tmp_path / "assistant.json", tmp_path, force=True)
    assert path.read_bytes() == original


def test_total_deadline_cancels_an_active_upstream_and_releases_the_queue(tmp_path) -> None:
    release = threading.Event()

    class StalledHandler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.flush()
            release.wait(3)

    backend = ThreadingHTTPServer(("127.0.0.1", 0), StalledHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    runtime = FakeRuntime()
    runtime.port = backend.server_port
    runtime.token = "test"
    broker = RequestBroker(config(tmp_path, request_timeout_seconds=0.05), runtime)
    server = ModelHTTPServer(broker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body, _ = call_http(
            server,
            "/v1/chat/completions",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert status == 504, body
        wait_for(lambda: not broker.status()["active"])
        assert broker.status()["queued"] == 0
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        broker.close()
        backend.shutdown()
        backend.server_close()
        thread.join(2)
        backend_thread.join(2)
