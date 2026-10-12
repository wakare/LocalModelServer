from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
from urllib.error import HTTPError
from urllib.parse import urlparse

from local_model_service import __version__
from local_model_service.config import is_loopback_endpoint


class ModelIdentityError(RuntimeError):
    pass


def describe_service(
    endpoint: str, model: str, api_key_env: str = "", *, follow_selected: bool = False
) -> dict:
    """Metadata reads never load a model. No proxies or HTTP redirects are followed."""
    if not is_loopback_endpoint(endpoint):
        raise ModelIdentityError("本地模型服务必须使用本机回环地址")
    parsed = urlparse(endpoint)
    connection_type = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=3)
    headers = {}
    key = os.environ.get(api_key_env, "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        connection.request("GET", "/v1/models", headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise ModelIdentityError(f"模型服务身份查询失败：HTTP {response.status}")
        body = json.loads(response.read(1024 * 1024))
        models = body.get("data", [])
        if len(models) != 1:
            raise ModelIdentityError("模型服务必须返回唯一模型身份")
        metadata = models[0].get("local_model_service", {})
        if metadata.get("service") != "local-model-service":
            raise ModelIdentityError("当前端点不是独立模型服务，请先启动 local-model-service")
        if not follow_selected and model not in {
            metadata.get("requested_model"),
            metadata.get("effective_model"),
        }:
            raise ModelIdentityError("助手模型名称与服务配置不一致")
        if (
            not isinstance(metadata.get("requested_model"), str)
            or not metadata["requested_model"]
            or models[0].get("id") != metadata["requested_model"]
            or metadata.get("effective_model")
            != metadata.get("descriptor", {}).get("effective_model")
        ):
            raise ModelIdentityError("模型服务返回的模型名称无效")
        fingerprint = hashlib.sha256(
            json.dumps(metadata["descriptor"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if fingerprint != metadata.get("identity"):
            raise ModelIdentityError("模型服务返回的身份指纹无效")
        return metadata
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as error:
        raise ModelIdentityError(f"无法确认本地模型服务身份：{error}") from error
    finally:
        connection.close()


def post_chat(endpoint: str, payload: bytes, headers: dict, timeout: float) -> tuple[dict, str]:
    if not is_loopback_endpoint(endpoint):
        raise ModelIdentityError("本地模型接口必须使用本机回环地址")
    parsed = urlparse(endpoint)
    connection_type = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    try:
        headers = {"User-Agent": f"local-model-service-client/{__version__}", **headers}
        connection.request("POST", parsed.path, body=payload, headers=headers)
        response = connection.getresponse()
        body = response.read(16 * 1024 * 1024 + 1)
        if response.status != 200:
            raise HTTPError(
                endpoint, response.status, response.reason, response.headers, io.BytesIO(body)
            )
        identity = response.getheader("X-Local-Model-Identity", "")
        if identity != headers.get("X-Expected-Model-Identity"):
            raise ModelIdentityError("模型响应身份与本次请求不一致")
        return json.loads(body), identity
    finally:
        connection.close()
