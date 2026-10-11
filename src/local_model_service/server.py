from __future__ import annotations

import hmac
import json
import logging
import os
import select
import socket
import threading
import time
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from local_model_service import __version__
from local_model_service.broker import RequestBroker, ServiceError
from local_model_service.catalog import model_catalog, select_model
from local_model_service.runtime import LocalModelServerError, ManagedRuntime


class ModelHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        broker: RequestBroker,
        *,
        dashboard_token: str = "",
        restart_handler: Callable[[], None] | None = None,
        config_path: Path | None = None,
        dashboard_command: str = "",
    ) -> None:
        self.broker = broker
        self.instance_id = uuid.uuid4().hex
        self.control_token = uuid.uuid4().hex
        self.dashboard_token = dashboard_token or (uuid.uuid4().hex + uuid.uuid4().hex)
        self.restart_handler = restart_handler
        self.config_path = config_path
        self.dashboard_command = dashboard_command
        self.service_action = ""
        self._action_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.started_at = time.time()
        self._handlers = threading.BoundedSemaphore(broker.config.queue_capacity + 8)
        if broker.config.api_key_env and not os.environ.get(broker.config.api_key_env):
            raise ValueError("配置的服务 API key 环境变量为空")
        super().__init__((broker.config.host, broker.config.port), ModelRequestHandler)

    def process_request(self, request, client_address) -> None:
        if not self._handlers.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n"
                    b"Connection: close\r\nRetry-After: 2\r\n\r\n"
                )
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._handlers.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._handlers.release()

    def server_bind(self) -> None:
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def metadata(self) -> dict:
        return {
            "service": "local-model-service",
            "version": __version__,
            "instance_id": self.instance_id,
            "service_pid": os.getpid(),
            "started_at": self.started_at,
            "request_timeout_seconds": self.broker.config.request_timeout_seconds,
            "service_action": self.service_action,
            "service_directory": str(self.config_path.parent) if self.config_path else "",
            "dashboard_command": self.dashboard_command,
            **self.broker.describe(),
        }


class ModelRequestHandler(BaseHTTPRequestHandler):
    server: ModelHTTPServer
    protocol_version = "HTTP/1.1"
    server_version = "LocalModelService/0.1"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)
        self._stream_started = False
        self._response_lock = threading.RLock()

    def log_message(self, format: str, *args) -> None:
        # Do not log prompts, authorization headers, or sentence text.
        logging.debug("HTTP %s", args[1] if len(args) > 1 else "request")

    def _authorize(self, *, dashboard: bool = False, control: bool = False) -> bool:
        self.close_connection = True
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin", "")
        try:
            address = urlsplit(f"http://{host}")
            local = (
                address.hostname in {"127.0.0.1", "localhost"}
                and (address.port or 80) == self.server.server_port
                and not address.username
                and not address.password
            )
        except ValueError:
            local = False
        if not local or (origin and (not dashboard or origin != f"http://{host}")):
            self._error(ServiceError("LOCAL_CLIENT_REQUIRED", "只接受本机应用调用", 403))
            return False
        if control:
            if not hmac.compare_digest(
                self.headers.get("X-Service-Control-Token", ""), self.server.dashboard_token
            ):
                self._error(ServiceError("UNAUTHORIZED", "请通过桌面模型服务控制台启用操作", 403))
                return False
            return True
        if dashboard:
            return True
        variable = self.server.broker.config.api_key_env
        key = os.environ.get(variable, "") if variable else ""
        if key and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {key}"):
            self._error(ServiceError("UNAUTHORIZED", "API key 无效", 401))
            return False
        return True

    def _headers(
        self,
        status: int,
        content_type: str,
        *,
        length: int | None = None,
        identity: str = "",
        effective_model: str = "",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        if content_type.startswith("text/html"):
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'; img-src data:; base-uri 'none'; "
                "frame-ancestors 'none'; form-action 'none'",
            )
        if length is not None:
            self.send_header("Content-Length", str(length))
        if identity:
            self.send_header("X-Local-Model-Identity", identity)
            self.send_header("X-Local-Model-Effective", effective_model)
        if status in {429, 503}:
            self.send_header("Retry-After", "2")
        self.end_headers()

    def _json(
        self, status: int, value: dict, *, identity: str = "", effective_model: str = ""
    ) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._headers(
            status,
            "application/json; charset=utf-8",
            length=len(body),
            identity=identity,
            effective_model=effective_model,
        )
        self.wfile.write(body)

    def _error(self, error: ServiceError) -> None:
        with self._response_lock:
            self._send_error(error)

    def _send_error(self, error: ServiceError) -> None:
        if self._stream_started:
            self.wfile.write(
                b"data: "
                + json.dumps({"error": {"code": error.code, "message": str(error)}}).encode()
                + b"\n\n"
            )
            self.wfile.flush()
        else:
            self._json(
                error.status,
                {
                    "error": {
                        "code": error.code,
                        "message": str(error),
                        "type": "local_model_service_error",
                    }
                },
            )

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        dashboard = path in {"/", "/dashboard", "/admin/status", "/admin/models"}
        if not self._authorize(dashboard=dashboard, control=path.startswith("/admin/")):
            return
        if path in {"/", "/dashboard"}:
            body = Path(__file__).with_name("dashboard.html").read_bytes()
            self._headers(200, "text/html; charset=utf-8", length=len(body))
            self.wfile.write(body)
            return
        if path == "/admin/models":
            try:
                with self.server._action_lock:
                    self._json(200, model_catalog(self.server.broker.config))
            except OSError as error:
                self._error(ServiceError("CATALOG_UNAVAILABLE", f"无法读取模型目录：{error}"))
            return
        metadata = self.server.metadata()
        if path in {"/health", "/v1/health", "/status", "/admin/status"}:
            self._json(200, metadata)
        elif path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": metadata["requested_model"],
                            "object": "model",
                            "created": int(self.server.started_at),
                            "owned_by": "local",
                            "local_model_service": metadata,
                        }
                    ],
                },
            )
        else:
            self._error(ServiceError("NOT_FOUND", "接口不存在", 404))

    def _client_closed(self) -> bool:
        readable, _, _ = select.select([self.connection], [], [], 0)
        if readable:
            try:
                return not self.connection.recv(1, socket.MSG_PEEK)
            except OSError:
                return True
        return False

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path.startswith("/admin/"):
            if self._authorize(dashboard=True, control=True):
                try:
                    self._admin_action(path)
                except ServiceError as error:
                    self._error(error)
            return
        if not self._authorize():
            return
        job = None
        try:
            if urlsplit(self.path).path == "/shutdown":
                if not hmac.compare_digest(
                    self.headers.get("X-Service-Control-Token", ""), self.server.control_token
                ):
                    raise ServiceError("UNAUTHORIZED", "服务控制凭证无效", 403)
                self._json(200, {"status": "stopping"})
                self.server.stop_event.set()
                return
            if urlsplit(self.path).path != "/v1/chat/completions":
                raise ServiceError("NOT_FOUND", "接口不存在", 404)
            if self.headers.get("Transfer-Encoding"):
                raise ServiceError("INVALID_REQUEST", "请使用 Content-Length 请求", 400)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ServiceError("INVALID_REQUEST", "Content-Length 无效", 400) from error
            if not 0 < length <= self.server.broker.config.max_body_bytes:
                raise ServiceError("INVALID_REQUEST", "请求体为空或超过限制", 413)
            try:
                payload = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError) as error:
                raise ServiceError("INVALID_REQUEST", "请求体必须是 JSON", 400) from error
            if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
                raise ServiceError("INVALID_REQUEST", "必须提供 model 字符串", 400)
            if not isinstance(payload.get("messages"), list) or not payload["messages"]:
                raise ServiceError("INVALID_REQUEST", "必须提供 messages 数组", 400)
            if not isinstance(payload.get("stream", False), bool):
                raise ServiceError("INVALID_REQUEST", "stream 必须是布尔值", 400)
            try:
                priority = int(self.headers.get("X-Request-Priority", "0"))
            except ValueError as error:
                raise ServiceError("INVALID_REQUEST", "请求优先级无效", 400) from error
            expected = self.headers.get("X-Expected-Model-Identity", "")
            if expected and (
                len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)
            ):
                raise ServiceError("INVALID_REQUEST", "模型身份格式无效", 400)
            job = self.server.broker.submit(
                payload,
                priority=priority,
                expected_identity=expected,
                stream=self._stream if payload.get("stream") else None,
            )
            while not job.done.wait(0.25):
                if self._client_closed():
                    job.cancel()
                    return
                if time.monotonic() >= job.deadline:
                    job.cancel()
                    raise ServiceError("REQUEST_TIMEOUT", "请求排队或执行超时", 504)
            if job.error:
                raise job.error
            if not self._stream_started:
                self._json(
                    200, job.result, identity=job.identity, effective_model=job.effective_model
                )
        except ServiceError as error:
            self._error(error)
        except OSError as error:
            if job:
                job.cancel()
            logging.debug("Client disconnected: %s", error)

    def _admin_action(self, path: str) -> None:
        if path not in {"/admin/stop", "/admin/restart", "/admin/unload", "/admin/model"}:
            raise ServiceError("NOT_FOUND", "操作不存在", 404)
        if self.headers.get("X-Expected-Service-Instance") != self.server.instance_id:
            raise ServiceError(
                "SERVICE_INSTANCE_CHANGED", "服务实例已变化，请刷新状态后再操作", 409
            )
        with self.server._action_lock:
            if self.server.service_action or self.server.stop_event.is_set():
                raise ServiceError("SERVICE_STOPPING", "服务正在停止或重启", 409)
            if path == "/admin/model":
                self._select_model()
                return
            if path == "/admin/unload":
                self.server.broker.unload_if_idle()
                self._json(200, {"status": "sleeping"})
                return
            if path == "/admin/restart" and not self.server.restart_handler:
                raise ServiceError("RESTART_UNAVAILABLE", "当前运行方式不支持网页重启", 503)
            self.server.broker.freeze_if_idle()
            self.server.service_action = path.rsplit("/", 1)[1]
            if path == "/admin/restart":
                try:
                    self.server.restart_handler()
                except (OSError, RuntimeError) as error:
                    self.server.service_action = ""
                    self.server.broker.resume_admission()
                    raise ServiceError(
                        "RESTART_FAILED", "无法启动重启程序，请检查服务日志", 503
                    ) from error
                self._json(200, {"status": "restarting"})
            else:
                self._json(200, {"status": "stopping"})
                self.server.stop_event.set()

    def _select_model(self) -> None:
        config_path = self.server.config_path
        if config_path is None:
            raise ServiceError("MODEL_SWITCH_UNAVAILABLE", "当前运行方式未提供模型配置文件", 503)
        if (
            self.headers.get("X-Expected-Model-Identity")
            != self.server.broker.runtime.describe()["identity"]
        ):
            raise ServiceError("MODEL_IDENTITY_CHANGED", "模型身份已变化，请刷新状态后再选择", 409)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.headers.get("Transfer-Encoding") or not 0 < length <= 2048:
                raise ValueError("模型选择请求无效")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
                raise ValueError("必须提供模型选项 id")
        except (ValueError, UnicodeDecodeError) as error:
            raise ServiceError("INVALID_REQUEST", str(error), 400) from error
        self.server.broker.freeze_if_idle()
        self.server.service_action = "switch"
        try:
            selected = select_model(self.server.broker.config, payload["id"])
            runtime = ManagedRuntime(selected, config_path.parent)
            self.server.broker.replace_runtime(
                selected, runtime, lambda: selected.save(config_path)
            )
        except (ValueError, LocalModelServerError) as error:
            raise ServiceError("MODEL_UNAVAILABLE", str(error), 400) from error
        except OSError as error:
            raise ServiceError(
                "MODEL_SWITCH_FAILED", "无法保存模型配置，请检查服务日志", 503
            ) from error
        finally:
            self.server.service_action = ""
            self.server.broker.resume_admission()
        self._json(200, {"status": "sleeping", "model": selected.model.model})

    def _stream(self, line: bytes | None, identity: str, model: str) -> None:
        with self._response_lock:
            self._send_stream(line, identity, model)

    def _send_stream(self, line: bytes | None, identity: str, model: str) -> None:
        if line is None:
            self._headers(200, "text/event-stream", identity=identity, effective_model=model)
            self._stream_started = True
        else:
            self.wfile.write(line)
            self.wfile.flush()
