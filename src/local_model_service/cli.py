from __future__ import annotations

import argparse
import base64
import http.client
import json
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import suppress
from dataclasses import fields
from logging.handlers import RotatingFileHandler
from pathlib import Path

from local_model_service.broker import RequestBroker
from local_model_service.config import ModelSettings, ServiceSettings, default_service_dir
from local_model_service.presets import infer_local_model_preset, local_model_preset
from local_model_service.runtime import ManagedRuntime, RuntimeFiles
from local_model_service.server import ModelHTTPServer


def _command(config_path: Path, action: str = "serve") -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--config", str(config_path), action]
    executable = Path(sys.executable)
    if os.name == "nt" and executable.with_name("pythonw.exe").is_file():
        executable = executable.with_name("pythonw.exe")
    return [str(executable), "-m", "local_model_service", "--config", str(config_path), action]


def _dashboard_command(config_path: Path) -> str:
    arguments = (
        "'" + value.replace("'", "''") + "'" for value in _command(config_path, "dashboard")
    )
    return "& " + " ".join(arguments)


def _http(
    config: ServiceSettings, path: str, *, method: str = "GET", headers=None, timeout: float = 3
) -> dict:
    connection = http.client.HTTPConnection(config.host, config.port, timeout=timeout)
    actual_headers = dict(headers or {})
    key = os.environ.get(config.api_key_env, "")
    if key:
        actual_headers["Authorization"] = f"Bearer {key}"
    try:
        connection.request(method, path, headers=actual_headers)
        response = connection.getresponse()
        body = json.loads(response.read(1024 * 1024))
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body}")
        if method == "GET" and body.get("service") != "local-model-service":
            raise RuntimeError("端口由其他服务占用")
        return body
    finally:
        connection.close()


def initialize(
    config_path: Path,
    assistant_path: Path,
    assets_dir: Path,
    *,
    force: bool = False,
    connect_assistant: bool = False,
) -> ServiceSettings:
    if config_path.exists() and not force:
        raise ValueError("服务配置已存在；修改前请停止服务，并显式使用 --force")
    if config_path.exists():
        current = ServiceSettings.load(config_path)
        try:
            _http(current, "/status")
        except ConnectionRefusedError:
            pass
        else:
            raise ValueError("请先停止正在运行的服务，再重新初始化配置")
    raw = json.loads(assistant_path.read_text(encoding="utf-8")) if assistant_path.is_file() else {}
    names = {field.name for field in fields(ModelSettings)}
    values = {key: value for key, value in raw.get("local_analysis", {}).items() if key in names}
    if values:
        values.setdefault(
            "model_preset",
            infer_local_model_preset(
                model=str(values.get("model", "")),
                model_path=str(values.get("model_path", "")),
            ),
        )
        preset = local_model_preset(values["model_preset"])
        values.setdefault("server_speculative_type", preset.speculative_type if preset else "none")
        values.setdefault("server_flash_attention", preset.flash_attention if preset else "auto")
        values.setdefault("server_draft_max_tokens", preset.draft_max_tokens if preset else 0)
    model = ModelSettings(**values)
    files = RuntimeFiles(model, assets_dir)
    model.server_executable = str(files.resolve_executable().resolve())
    model.model_path = str(files.resolve_model_path().resolve())
    if model.server_speculative_type != "none":
        try:
            model.draft_model_path = str(files.resolve_draft_model_path().resolve())
        except RuntimeError:
            if not model.server_draft_fallback:
                raise
    config = ServiceSettings(assets_dir=str(assets_dir.resolve()), model=model)
    if config_path.exists():
        _backup(config_path)
    config.save(config_path)
    if connect_assistant:
        _connect_assistant(config, assistant_path)
    return config


def _connect_assistant(config: ServiceSettings, assistant_path: Path) -> None:
    raw = json.loads(assistant_path.read_text(encoding="utf-8")) if assistant_path.is_file() else {}
    if assistant_path.exists():
        _backup(assistant_path)
    raw.setdefault("local_analysis", {}).update(
        manage_server=False,
        endpoint=f"http://{config.host}:{config.port}/v1/chat/completions",
        model=config.model.model,
        timeout_seconds=int(config.request_timeout_seconds) + 10,
        runtime_identity="",
    )
    assistant_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = assistant_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(assistant_path)


def _backup(path: Path) -> None:
    backup = path.with_name(f"{path.name}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
    if backup.exists():
        raise ValueError(f"备份已存在：{backup}")
    backup.write_bytes(path.read_bytes())


def dashboard_token(service_dir: Path) -> str:
    path = service_dir / "dashboard.key"
    try:
        token = path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        token = secrets.token_hex(32)
        with path.open("x", encoding="ascii") as output:
            output.write(token)
        if os.name != "nt":
            path.chmod(0o600)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("控制台凭证文件无效，请检查 dashboard.key")
    return token


def _restart_helper(config_path: Path) -> None:
    with (config_path.parent / "launcher.log").open("ab") as output:
        subprocess.Popen(
            _command(config_path, "restart"),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )


def serve(config: ServiceSettings, config_path: Path) -> None:
    service_dir = config_path.parent
    service_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        service_dir / "service.log", maxBytes=1024 * 1024, backupCount=3, encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    runtime = ManagedRuntime(config, service_dir)
    broker = RequestBroker(config, runtime)
    server = None
    control_path = service_dir / "control.json"
    try:
        server = ModelHTTPServer(
            broker,
            dashboard_token=dashboard_token(service_dir),
            restart_handler=lambda: _restart_helper(config_path),
            config_path=config_path,
            dashboard_command=_dashboard_command(config_path),
        )
        control_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "instance_id": server.instance_id,
                    "token": server.control_token,
                }
            ),
            encoding="utf-8",
        )
        if os.name != "nt":
            control_path.chmod(0o600)
        for kind in (signal.SIGINT, signal.SIGTERM):
            signal.signal(kind, lambda *_: server.stop_event.set())
        thread = threading.Thread(target=server.serve_forever, name="model-http", daemon=True)
        thread.start()
        logging.info(
            "Listening on %s:%s; model stays unloaded until an inference request",
            *server.server_address,
        )
        server.stop_event.wait()
        server.shutdown()
        thread.join(timeout=5)
    finally:
        broker.close()
        if server:
            server.server_close()
            if control_path.exists():
                control = json.loads(control_path.read_text(encoding="utf-8"))
                if control.get("instance_id") == server.instance_id:
                    control_path.unlink()
        logging.info("Service stopped")


def start(config: ServiceSettings, config_path: Path) -> dict:
    try:
        return _http(config, "/status")
    except (OSError, http.client.HTTPException):
        pass
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with (config_path.parent / "launcher.log").open("ab") as output:
        process = subprocess.Popen(
            _command(config_path),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"服务启动失败（{process.returncode}），请检查 service.log/launcher.log"
            )
        try:
            return _http(config, "/status")
        except (OSError, http.client.HTTPException):
            time.sleep(0.2)
    raise RuntimeError("服务未在 10 秒内开始监听，请检查日志")


def _port_available(config: ServiceSettings) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind((config.host, config.port))
        except OSError:
            return False
        return True


def stop(config: ServiceSettings, config_path: Path) -> None:
    # On Windows a closed loopback port may time out instead of refusing connects.
    # Verify exclusive bind availability before allowing the replacement to start.
    if _port_available(config):
        return
    status = _http(config, "/status")
    control_path = config_path.parent / "control.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    if control["instance_id"] != status["instance_id"]:
        raise RuntimeError("服务控制文件与当前实例不一致")
    _http(
        config,
        "/shutdown",
        method="POST",
        headers={"X-Service-Control-Token": control["token"]},
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if _port_available(config):
            return
        try:
            current = _http(config, "/status", timeout=0.3)
        except (OSError, http.client.HTTPException):
            time.sleep(0.1)
            continue
        if current["instance_id"] != status["instance_id"]:
            raise RuntimeError("停止期间服务实例已变化，请重新查询状态")
        time.sleep(0.1)
    raise RuntimeError("服务未在 20 秒内完成停止，请检查 service.log")


def restart(config: ServiceSettings, config_path: Path) -> dict:
    stop(config, config_path)
    return start(config, config_path)


def open_dashboard(config: ServiceSettings, config_path: Path, *, browser: bool = True) -> None:
    start(config, config_path)
    token = dashboard_token(config_path.parent)
    url = f"http://{config.host}:{config.port}/dashboard"
    if browser and not webbrowser.open(f"{url}#{token}"):
        raise RuntimeError("无法打开默认浏览器，请使用桌面模型服务控制台入口")
    print(f"模型服务控制台：{url}")


def install_shortcut(config_path: Path) -> None:
    if os.name != "nt":
        raise ValueError("桌面控制台入口仅适用于 Windows")

    def literal(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    command = _command(config_path, "dashboard")
    script = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "$modelServiceLinkPath = Join-Path ([Environment]::GetFolderPath('Desktop')) "
            "'本机模型服务.lnk'",
            "$modelServiceShell = New-Object -ComObject WScript.Shell",
            "$modelServiceLink = $modelServiceShell.CreateShortcut($modelServiceLinkPath)",
            "if ((Test-Path -LiteralPath $modelServiceLinkPath) -and "
            f"($modelServiceLink.TargetPath -ne {literal(command[0])})) "
            "{ throw '已有同名快捷方式指向其他程序，请先调整名称。' }",
            f"$modelServiceLink.TargetPath = {literal(command[0])}",
            f"$modelServiceLink.Arguments = {literal(subprocess.list2cmdline(command[1:]))}",
            f"$modelServiceLink.WorkingDirectory = {literal(str(config_path.parent))}",
            "$modelServiceLink.Description = '开启本机模型服务并打开控制台'",
            "$modelServiceLink.Save()",
        ]
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-EncodedCommand", encoded],
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def startup(config_path: Path, *, remove: bool = False) -> None:
    if os.name != "nt":
        raise ValueError("登录自启注册仅适用于 Windows")
    import winreg

    with winreg.CreateKey(
        winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"
    ) as key:
        if remove:
            with suppress(FileNotFoundError):
                winreg.DeleteValue(key, "LocalModelService")
        else:
            winreg.SetValueEx(
                key,
                "LocalModelService",
                0,
                winreg.REG_SZ,
                subprocess.list2cmdline(_command(config_path, "serve")),
            )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:] or ["dashboard"]
    parser = argparse.ArgumentParser(description="独立本机模型服务（按需加载、空闲卸载）")
    parser.add_argument("--config", type=Path, default=default_service_dir() / "service.json")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="从助手配置初始化独立服务配置")
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "KindleAnkiBridge"
    init.add_argument("--assistant-settings", type=Path, default=base / "settings.json")
    init.add_argument("--assets-dir", type=Path, default=base)
    init.add_argument("--force", action="store_true")
    init.add_argument("--connect-assistant", action="store_true")
    connect = commands.add_parser("connect-assistant", help="将助手连接到已配置的独立服务")
    connect.add_argument("--assistant-settings", type=Path, default=base / "settings.json")
    dashboard = commands.add_parser("dashboard", help="开启服务并打开控制台")
    dashboard.add_argument("--no-browser", action="store_true")
    for command in (
        "serve",
        "start",
        "stop",
        "restart",
        "status",
        "install-shortcut",
        "install-startup",
        "uninstall-startup",
    ):
        commands.add_parser(command)
    arguments = parser.parse_args(argv)
    config_path = arguments.config.expanduser().resolve()
    try:
        if arguments.command == "init":
            initialize(
                config_path,
                arguments.assistant_settings,
                arguments.assets_dir,
                force=arguments.force,
                connect_assistant=arguments.connect_assistant,
            )
            print(str(config_path))
            return 0
        config = ServiceSettings.load(config_path)
        if arguments.command == "serve":
            serve(config, config_path)
        elif arguments.command == "start":
            print(json.dumps(start(config, config_path), ensure_ascii=False, indent=2))
        elif arguments.command == "restart":
            print(json.dumps(restart(config, config_path), ensure_ascii=False, indent=2))
        elif arguments.command == "dashboard":
            open_dashboard(config, config_path, browser=not arguments.no_browser)
        elif arguments.command == "install-shortcut":
            install_shortcut(config_path)
        elif arguments.command == "status":
            print(json.dumps(_http(config, "/status"), ensure_ascii=False, indent=2))
        elif arguments.command == "connect-assistant":
            _connect_assistant(config, arguments.assistant_settings)
        elif arguments.command == "stop":
            stop(config, config_path)
        else:
            startup(config_path, remove=arguments.command == "uninstall-startup")
        return 0
    except (OSError, ValueError, RuntimeError, http.client.HTTPException) as error:
        if sys.stderr is None:
            with (
                suppress(OSError),
                (config_path.parent / "launcher.log").open("a", encoding="utf-8") as output,
            ):
                output.write(f"{arguments.command}: {error}\n")
        else:
            print(str(error), file=sys.stderr)
        return 2
