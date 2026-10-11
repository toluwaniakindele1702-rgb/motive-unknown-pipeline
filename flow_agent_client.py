"""Relic Loop integration client for the local Flow Agent REST server.

This module is intentionally opt-in. It lets the existing Relic Loop pipeline
use Google Flow through kodelyx/flow-agent without changing the normal provider
chain until FLOW_AGENT_ENABLED=1 is set.

Flow Agent runs on the machine that owns the authenticated Google Flow session.
The Relic Loop GitHub runner therefore needs network access to FLOW_AGENT_URL;
127.0.0.1 only works when the pipeline itself is running on that same machine.
"""
from __future__ import annotations

import base64
import os
import shutil
from pathlib import Path
from typing import Any

import requests


FLOW_AGENT_URL = os.environ.get("FLOW_AGENT_URL", "http://127.0.0.1:8001").rstrip("/")
FLOW_AGENT_TIMEOUT = float(os.environ.get("FLOW_AGENT_TIMEOUT", "300"))
FLOW_AGENT_IMAGE_MODEL = os.environ.get("FLOW_AGENT_IMAGE_MODEL", "narwhal").strip()
FLOW_AGENT_VIDEO_DURATION = os.environ.get("FLOW_AGENT_VIDEO_DURATION", "8s").strip()
FLOW_AGENT_VIDEO_QUALITY = os.environ.get("FLOW_AGENT_VIDEO_QUALITY", "720p").strip()


def enabled() -> bool:
    return os.environ.get("FLOW_AGENT_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def health() -> dict[str, Any]:
    response = requests.get(f"{FLOW_AGENT_URL}/health", timeout=20)
    response.raise_for_status()
    return response.json()


def balance() -> dict[str, Any]:
    response = requests.get(f"{FLOW_AGENT_URL}/api/v1/balance", timeout=30)
    response.raise_for_status()
    return response.json()


def _download_media(url: str, out_path: Path) -> None:
    # Flow Agent normally returns localhost URLs. If the URL is absolute, keep
    # only the path and fetch it from our configured Flow Agent host so a
    # response cannot accidentally redirect the pipeline to another host.
    parsed_path = url.split("/api/v1/media/", 1)[-1].split("/v1/media/", 1)[-1]
    if parsed_path == url:
        raise RuntimeError(f"Unexpected Flow Agent media URL: {url}")
    response = requests.get(
        f"{FLOW_AGENT_URL}/api/v1/media/{parsed_path}",
        timeout=FLOW_AGENT_TIMEOUT,
    )
    response.raise_for_status()
    raw = response.content
    if len(raw) < 1000:
        raise RuntimeError("Flow Agent returned an empty/tiny media payload.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)


def _download_file_entry(entry: dict[str, Any], out_path: Path) -> None:
    url = str(entry.get("url", "")).strip()
    path = str(entry.get("path", "")).strip()
    if url:
        _download_media(url, out_path)
        return
    if path and Path(path).exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_path)
        return
    b64 = entry.get("b64_json")
    if isinstance(b64, str) and b64:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(base64.b64decode(b64))
        return
    raise RuntimeError("Flow Agent response contained no downloadable media.")


def generate_image(prompt: str, out_path: Path, *, aspect: str = "16:9", count: int = 1) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "aspect": aspect,
        "count": max(1, min(4, int(count))),
        "model": FLOW_AGENT_IMAGE_MODEL,
    }
    response = requests.post(
        f"{FLOW_AGENT_URL}/api/v1/image",
        json=payload,
        timeout=FLOW_AGENT_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    files = data.get("files") or []
    if not files:
        raise RuntimeError(f"Flow Agent image response contained no files: {data}")
    _download_file_entry(files[0], out_path)
    return data


def generate_video(
    prompt: str,
    out_path: Path,
    *,
    aspect: str = "16:9",
    duration: str | None = None,
    quality: str | None = None,
    start_image: Path | str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "aspect": aspect,
        "duration": duration or FLOW_AGENT_VIDEO_DURATION,
        "quality": quality or FLOW_AGENT_VIDEO_QUALITY,
        "count": 1,
    }
    if start_image:
        payload["start_image"] = str(start_image)
    response = requests.post(
        f"{FLOW_AGENT_URL}/api/v1/video",
        json=payload,
        timeout=FLOW_AGENT_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    files = data.get("files") or []
    if not files:
        raise RuntimeError(f"Flow Agent video response contained no files: {data}")
    _download_file_entry(files[0], out_path)
    return data


def diagnostics() -> dict[str, Any]:
    result: dict[str, Any] = {"url": FLOW_AGENT_URL}
    try:
        result["health"] = health()
    except Exception as exc:
        result["health_error"] = str(exc)
        return result
    try:
        result["balance"] = balance()
    except Exception as exc:
        result["balance_error"] = str(exc)
    return result
