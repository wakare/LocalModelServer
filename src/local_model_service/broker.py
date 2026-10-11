from __future__ import annotations

import http.client
import json
import logging
import socket
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field

from local_model_service.config import ServiceSettings
from local_model_service.runtime import LocalModelServerError, ManagedRuntime


class ServiceError(RuntimeError):
    def __init__(self, code: str, message: str, status: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass
class RequestJob:
    payload: dict
    priority: int
    sequence: int
    deadline: float
    expected_identity: str = ""
    stream: Callable[[bytes | None, str, str], None] | None = None
    submitted: float = field(default_factory=time.monotonic)
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: Exception | None = None
    identity: str = ""
    effective_model: str = ""
    connection: http.client.HTTPConnection | None = None
    upstream_socket: socket.socket | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def cancel(self) -> None:
        self.cancelled.set()
        with self.lock:
            if self.upstream_socket:
                with suppress(OSError):
                    self.upstream_socket.shutdown(socket.SHUT_RDWR)


class RequestBroker:
    """One event-driven worker, a bounded queue, and no inference while idle."""

    def __init__(self, config: ServiceSettings, runtime: ManagedRuntime) -> None:
        self.config = config
        self.runtime = runtime
        self._condition = threading.Condition()
        self._pending: list[RequestJob] = []
        self._active: RequestJob | None = None
        self._sequence = 0
        self._closing = False
        self._accepting = True
        self._last_completed = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="inference-queue", daemon=True)
        self._thread.start()

    def submit(
        self,
        payload: dict,
        *,
        priority: int = 0,
        expected_identity: str = "",
        stream: Callable | None = None,
    ) -> RequestJob:
        with self._condition:
            if self._closing or not self._accepting:
                raise ServiceError("SERVICE_STOPPING", "服务正在停止")
            model = payload.get("model")
            if model not in {self.config.model.model, self.runtime.effective_alias}:
                raise ServiceError("MODEL_NOT_FOUND", "请求模型不属于当前服务配置", 404)
            retained = []
            for pending in self._pending:
                if pending.cancelled.is_set() or time.monotonic() >= pending.deadline:
                    pending.error = ServiceError("REQUEST_EXPIRED", "请求已取消或排队超时", 504)
                    pending.done.set()
                else:
                    retained.append(pending)
            self._pending = retained
            if len(self._pending) >= self.config.queue_capacity:
                raise ServiceError("QUEUE_FULL", "模型请求队列已满", 429)
            self._sequence += 1
            job = RequestJob(
                payload,
                max(-10, min(10, priority)),
                self._sequence,
                time.monotonic() + self.config.request_timeout_seconds,
                expected_identity=expected_identity,
                stream=stream,
            )
            self._pending.append(job)
            self._condition.notify()
            return job

    def status(self) -> dict:
        with self._condition:
            return {
                "queued": len(self._pending),
                "active": self._active is not None,
                "accepting_requests": self._accepting and not self._closing,
                "queue_capacity": self.config.queue_capacity,
                "idle_seconds": self.config.idle_seconds,
                "idle_unload_in_seconds": max(
                    0,
                    round(self.config.idle_seconds - (time.monotonic() - self._last_completed), 2),
                )
                if self.runtime.process and self._active is None
                else None,
            }

    def freeze_if_idle(self) -> None:
        with self._condition:
            if self._active or self._pending:
                raise ServiceError("SERVICE_BUSY", "还有请求正在执行或排队，请完成后再操作", 409)
            if not self._accepting or self._closing:
                raise ServiceError("SERVICE_STOPPING", "服务正在停止或重启", 409)
            self._accepting = False

    def resume_admission(self) -> None:
        with self._condition:
            if not self._closing:
                self._accepting = True

    def describe(self) -> dict:
        with self._condition:
            return {**self.runtime.describe(), **self.status()}

    def replace_runtime(
        self, config: ServiceSettings, runtime: ManagedRuntime, persist: Callable[[], None]
    ) -> None:
        with self._condition:
            if self._accepting or self._closing or self._active or self._pending:
                raise ServiceError("SERVICE_BUSY", "模型切换需要暂停接收请求且队列为空", 409)
            self.runtime.stop()
            # If persistence fails, the original unloaded model remains usable.
            persist()
            self.config, self.runtime = config, runtime
            self._last_completed = time.monotonic()
            self._condition.notify_all()
            logging.info("Model selected: %s; backend remains unloaded", config.model.model)

    def unload_if_idle(self) -> None:
        with self._condition:
            if self._active or self._pending:
                raise ServiceError(
                    "SERVICE_BUSY", "还有请求正在执行或排队，请完成后再释放模型", 409
                )
            self.runtime.stop()
            self._condition.notify_all()
            logging.info("Backend unloaded by dashboard")

    def close(self) -> None:
        with self._condition:
            self._closing = True
            for job in self._pending:
                job.cancel()
                job.error = ServiceError("SERVICE_STOPPING", "服务正在停止")
                job.done.set()
            self._pending.clear()
            if self._active:
                self._active.cancel()
            self._condition.notify_all()
        self._thread.join(timeout=12)
        if self._thread.is_alive():
            raise RuntimeError("推理工作线程未能停止")

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._pending and not self._closing:
                        if self.runtime.process is None:
                            self._condition.wait()
                        else:
                            remaining = self.config.idle_seconds - (
                                time.monotonic() - self._last_completed
                            )
                            if remaining <= 0:
                                self.runtime.stop()
                                logging.info("Idle timeout: backend unloaded")
                            else:
                                self._condition.wait(remaining)
                    if self._closing:
                        return
                    now = time.monotonic()
                    # Aging lets background jobs eventually outrank new interactive jobs.
                    job = min(
                        self._pending,
                        key=lambda j: (-j.priority - int((now - j.submitted) / 5), j.sequence),
                    )
                    self._pending.remove(job)
                    self._active = job
                try:
                    if job.cancelled.is_set() or time.monotonic() >= job.deadline:
                        raise ServiceError("REQUEST_EXPIRED", "请求已取消或排队超时", 504)
                    self._execute(job)
                except ServiceError as error:
                    job.error = error
                    if error.status >= 500 and error.code not in {"REQUEST_EXPIRED"}:
                        self.runtime.mark_failed(str(error))
                except TimeoutError as error:
                    job.error = ServiceError("INFERENCE_TIMEOUT", "模型推理超时", 504)
                    self.runtime.mark_failed(str(error))
                except (
                    OSError,
                    ValueError,
                    http.client.HTTPException,
                    LocalModelServerError,
                ) as error:
                    job.error = ServiceError("BACKEND_UNAVAILABLE", str(error))
                    self.runtime.mark_failed(str(error))
                    logging.warning("Inference failed: %s", error)
                except Exception as error:
                    job.error = ServiceError("INTERNAL_ERROR", str(error), 500)
                    self.runtime.mark_failed(str(error))
                    logging.exception("Inference worker error")
                finally:
                    job.done.set()
                    with self._condition:
                        self._active = None
                        self._last_completed = time.monotonic()
        finally:
            self.runtime.stop()

    def _execute(self, job: RequestJob) -> None:
        self.runtime.ensure_ready(job.cancelled, job.deadline)
        job.identity = self.runtime.identity
        job.effective_model = self.runtime.effective_alias
        if job.expected_identity and job.expected_identity != job.identity:
            raise ServiceError(
                "MODEL_IDENTITY_CHANGED", "实际模型身份已变化，请重新获取模型信息", 409
            )
        if job.cancelled.is_set() or time.monotonic() >= job.deadline:
            raise ServiceError("REQUEST_EXPIRED", "请求已取消或超时", 504)
        payload = {**job.payload, "model": job.effective_model}
        timeout = min(self.config.inference_timeout_seconds, job.deadline - time.monotonic())
        connection = http.client.HTTPConnection("127.0.0.1", self.runtime.port, timeout=timeout)
        with job.lock:
            job.connection = connection
        response = None
        try:
            connection.connect()
            with job.lock:
                job.upstream_socket = connection.sock
            if job.cancelled.is_set():
                raise ServiceError("REQUEST_CANCELLED", "请求已取消", 499)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.runtime.token}",
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                body = response.read(64 * 1024).decode("utf-8", errors="replace")
                if response.status >= 500:
                    self.runtime.stop()
                raise ServiceError("BACKEND_REQUEST_FAILED", body, response.status)
            if job.stream:
                if "text/event-stream" not in response.getheader("Content-Type", ""):
                    raise ServiceError("INVALID_BACKEND_RESPONSE", "backend 未返回 SSE", 502)
                job.stream(None, job.identity, job.effective_model)
                completed = False
                while not job.cancelled.is_set():
                    line = response.readline(1024 * 1024)
                    if not line:
                        break
                    job.stream(line, job.identity, job.effective_model)
                    if line.strip() == b"data: [DONE]":
                        completed = True
                if not completed and not job.cancelled.is_set():
                    raise ServiceError("INCOMPLETE_STREAM", "模型响应流意外中断", 502)
            else:
                body = response.read(16 * 1024 * 1024 + 1)
                if len(body) > 16 * 1024 * 1024:
                    raise ServiceError("INVALID_BACKEND_RESPONSE", "backend 响应过大", 502)
                job.result = json.loads(body)
                if (
                    not isinstance(job.result, dict)
                    or job.result.get("model") != job.effective_model
                ):
                    raise ServiceError("INVALID_BACKEND_IDENTITY", "backend 响应模型身份错误", 502)
        finally:
            with job.lock:
                job.connection = None
                job.upstream_socket = None
            if response:
                response.close()
            connection.close()
