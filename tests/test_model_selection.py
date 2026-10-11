from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from local_model_service.broker import RequestBroker
from local_model_service.catalog import model_catalog, select_model
from local_model_service.config import ModelSettings, ServiceSettings
from local_model_service.runtime import ManagedRuntime
from local_model_service.server import ModelHTTPServer


@pytest.fixture
def gateway(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "gemma-4-26B_q4_0-it.gguf").write_bytes(b"gemma main weight")
    main = models / "another-model.gguf"
    main.write_bytes(b"extra main weight")
    executable = tmp_path / "llama-server.exe"
    executable.write_bytes(b"dummy runtime, never launched")
    settings = ServiceSettings(
        port=0,
        assets_dir=str(tmp_path),
        model=ModelSettings(
            model="another-model",
            model_preset="custom",
            model_path=str(main),
            server_executable=str(executable),
            server_speculative_type="none",
        ),
    )
    runtime = ManagedRuntime(settings, tmp_path)
    broker = RequestBroker(settings, runtime)
    config_path = tmp_path / "service.json"
    server = ModelHTTPServer(broker, config_path=config_path)
    settings.port = server.server_port
    settings.save(config_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, broker, runtime, config_path
    finally:
        server.shutdown()
        server.server_close()
        broker.close()
        thread.join(2)


def headers(server) -> dict:
    return {
        "Origin": f"http://127.0.0.1:{server.server_port}",
        "X-Service-Control-Token": server.dashboard_token,
        "X-Expected-Service-Instance": server.instance_id,
        "X-Expected-Model-Identity": server.broker.runtime.describe()["identity"],
    }


def call(server, path, payload=None, ticket=True):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request(
            "GET" if payload is None else "POST",
            path,
            body=json.dumps(payload) if payload is not None else None,
            headers=headers(server) if ticket is True else ticket or {},
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_catalog_lists_installed_presets_and_extra_main_files_without_loading(gateway) -> None:
    server, broker, runtime, config_path = gateway
    models_dir = Path(broker.config.assets_dir) / "models"
    (models_dir / "Qwen3.8-27B-Q4_K_M.gguf").write_bytes(b"qwen main")
    (models_dir / "extra-DFlash2.gguf").write_bytes(b"not an instruction model")
    assert call(server, "/admin/models", ticket=False)[0] == 403
    status, catalog = call(server, "/admin/models")
    assert status == 200
    entries = {entry["id"]: entry for entry in catalog["models"]}
    assert entries["gemma4_26b_q4_0_it"]["available"]
    assert not entries["gemma4_31b_q4_0_it"]["available"]
    assert entries["gemma4_31b_q4_0_it"]["missing"] == ["gemma-4-31B_q4_0-it.gguf"]
    assert "草稿文件" in entries["qwen38_27b_q4_k_m_dflash2"]["warning"]
    assert entries["file:another-model.gguf"]["available"]
    assert "file:extra-DFlash2.gguf" not in entries
    assert catalog["current"] == "file:another-model.gguf"
    with pytest.raises(ValueError, match="已安装"):
        select_model(broker.config, "file:../outside.gguf")
    assert runtime.process is None
    assert ServiceSettings.load(config_path).model.model == "another-model"


def test_switch_busy_guard_preserves_active_and_queued_requests(gateway) -> None:
    server, broker, original, config_path = gateway
    entered, release = threading.Event(), threading.Event()
    initial_config = config_path.read_bytes()

    def execute(job):
        entered.set()
        release.wait(2)
        job.result = {}

    broker._execute = execute
    try:
        active = broker.submit({"model": "another-model"})
        assert entered.wait(2)
        queued = broker.submit({"model": "another-model"})
        status, body = call(server, "/admin/model", {"id": "gemma4_26b_q4_0_it"})
        assert status == 409 and body["error"]["code"] == "SERVICE_BUSY"
        assert broker.runtime is original and config_path.read_bytes() == initial_config
        assert not active.cancelled.is_set() and not queued.cancelled.is_set()
        release.set()
        assert active.done.wait(2) and queued.done.wait(2)
    finally:
        release.set()


@pytest.mark.parametrize("selection", ["unknown", "gemma4_31b_q4_0_it", "file:../model.gguf"])
def test_invalid_or_missing_model_keeps_original_config_and_admission(gateway, selection) -> None:
    server, broker, original, config_path = gateway
    initial_config = config_path.read_bytes()
    status, _ = call(server, "/admin/model", {"id": selection})
    assert status == 400
    assert broker.runtime is original and config_path.read_bytes() == initial_config
    assert broker.status()["accepting_requests"] and server.service_action == ""


def test_switch_write_failure_restores_original_model_and_acceptance(gateway, monkeypatch) -> None:
    server, broker, original, config_path = gateway
    initial_config = config_path.read_bytes()

    def fail(*_):
        raise OSError("read-only config")

    monkeypatch.setattr(ServiceSettings, "save", fail)
    status, body = call(server, "/admin/model", {"id": "gemma4_26b_q4_0_it"})
    assert status == 503 and body["error"]["code"] == "MODEL_SWITCH_FAILED"
    assert broker.runtime is original and config_path.read_bytes() == initial_config
    assert broker.status()["accepting_requests"] and server.service_action == ""


def test_extra_local_gguf_selection_clears_previous_draft_config(gateway) -> None:
    server, broker, _, config_path = gateway
    status, _ = call(server, "/admin/model", {"id": "gemma4_26b_q4_0_it"})
    assert status == 200
    status, _ = call(server, "/admin/model", {"id": "file:another-model.gguf"})
    assert status == 200
    saved = ServiceSettings.load(config_path)
    assert saved.model.model == "another-model" and saved.model.model_preset == "custom"
    assert saved.model.server_speculative_type == "none"
    assert model_catalog(broker.config)["current"] == "file:another-model.gguf"
    assert broker.runtime.process is None
