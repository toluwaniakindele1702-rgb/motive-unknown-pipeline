"""Chispa-Pifia multi-provider image router.

Priority: Pollinations (reference-guided edits) -> OVHcloud SDXL (prompt-only)
-> Cloudflare Workers AI (reference-guided final fallback).
The existing renderer retains storyboard, continuity, OCR, and manifest handling.
"""
from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

import cloudflare_relic_loop_images as base

POLLINATIONS_KEY = os.environ.get("POLLINATIONS_API_KEY", "").strip()
POLLINATIONS_MODEL = os.environ.get(
    "POLLINATIONS_IMAGE_MODEL", "bytedance/seedream-4.0"
).strip()
POLLINATIONS_URL = "https://gen.pollinations.ai/v1/images/edits"

OVH_TOKEN = os.environ.get("OVH_AI_ENDPOINTS_ACCESS_TOKEN", "").strip()
OVH_MODEL = os.environ.get("OVH_IMAGE_MODEL", "stable-diffusion-xl-base-v10").strip()
OVH_URL = os.environ.get(
    "OVH_IMAGE_ENDPOINT",
    "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/images/generations",
).strip()

PROVIDER_EVENTS: list[dict] = []
PROVIDER_COUNTS: dict[str, int] = {"pollinations": 0, "ovhcloud": 0, "cloudflare": 0}
REFERENCE_FILES = base.CANONICAL_REFERENCE_PATHS
_CLOUDFLARE_ORIGINAL = base._cloudflare


def _record(provider: str, status: str, detail: str = "") -> None:
    PROVIDER_EVENTS.append({
        "provider": provider,
        "status": status,
        "detail": detail[:700],
    })
    print(f"[PROVIDER:{provider.upper()}] {status}" + (f": {detail[:500]}" if detail else ""))


def _normalize_image(blob: bytes) -> bytes:
    """Normalize every provider output to the renderer's required vertical canvas."""
    try:
        with Image.open(BytesIO(blob)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            target_ratio = base.WIDTH / base.HEIGHT
            current_ratio = image.width / image.height
            if current_ratio > target_ratio:
                new_width = max(1, round(image.height * target_ratio))
                left = (image.width - new_width) // 2
                image = image.crop((left, 0, left + new_width, image.height))
            elif current_ratio < target_ratio:
                new_height = max(1, round(image.width / target_ratio))
                top = (image.height - new_height) // 2
                image = image.crop((0, top, image.width, top + new_height))
            image = image.resize((base.WIDTH, base.HEIGHT), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="PNG", optimize=True)
            return output.getvalue()
    except Exception as exc:
        raise RuntimeError(f"Provider returned an unreadable image: {type(exc).__name__}: {exc}") from exc


def _extract_image(response: requests.Response) -> bytes:
    content_type = response.headers.get("content-type", "").lower()
    if "image/" in content_type:
        return response.content
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Image API returned non-image, non-JSON response: {response.text[:500]}") from exc
    data = payload.get("data") or []
    if not data or not isinstance(data[0], dict):
        raise RuntimeError(f"Image API response contains no data[0]: {json.dumps(payload)[:700]}")
    item = data[0]
    encoded = item.get("b64_json")
    if encoded:
        try:
            return base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise RuntimeError("Image API returned invalid b64_json") from exc
    url = item.get("url")
    if url:
        downloaded = requests.get(url, timeout=90)
        downloaded.raise_for_status()
        return downloaded.content
    raise RuntimeError(f"Image API response has neither b64_json nor url: {json.dumps(item)[:500]}")


def _pollinations(prompt: str, refs: list[tuple[str, bytes, str]] | None) -> bytes:
    if not POLLINATIONS_KEY:
        raise RuntimeError("POLLINATIONS_API_KEY is not configured")
    if not refs:
        raise RuntimeError("Pollinations reference-edit mode requires canonical reference images")
    headers = {"Authorization": f"Bearer {POLLINATIONS_KEY}"}
    files = []
    for idx, (name, blob, mime) in enumerate(refs[:3]):
        files.append(("image", (f"reference_{idx}.jpg", blob, mime or "image/jpeg")))
    data = {
        "model": POLLINATIONS_MODEL,
        "prompt": prompt,
        "size": f"{base.WIDTH}x{base.HEIGHT}",
        "response_format": "b64_json",
        "n": "1",
    }
    last_error = None
    for attempt in range(1, 3):
        try:
            print(
                f"[POLLINATIONS] edit attempt {attempt}/2 model={POLLINATIONS_MODEL} "
                f"references={len(files)} size={base.WIDTH}x{base.HEIGHT}"
            )
            response = requests.post(
                POLLINATIONS_URL, headers=headers, data=data, files=files, timeout=240
            )
            if not response.ok:
                message = response.text[:1000]
                raise RuntimeError(f"HTTP {response.status_code}: {message}")
            return _normalize_image(_extract_image(response))
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(5 * attempt)
        except RuntimeError as exc:
            last_error = str(exc)
            status = getattr(locals().get("response"), "status_code", 0)
            if status not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                break
            time.sleep(5 * attempt)
    raise RuntimeError(last_error or "Pollinations image edit failed")


def _ovhcloud(prompt: str) -> bytes:
    if not OVH_TOKEN:
        raise RuntimeError("OVH_AI_ENDPOINTS_ACCESS_TOKEN is not configured")
    headers = {"Authorization": f"Bearer {OVH_TOKEN}"}
    payload = {
        "model": OVH_MODEL,
        "prompt": prompt,
        "size": "1024x1024",
        "n": 1,
        "response_format": "b64_json",
    }
    last_error = None
    for attempt in range(1, 3):
        try:
            print(f"[OVHCLOUD] generation attempt {attempt}/2 model={OVH_MODEL} (text-to-image; no reference upload)")
            response = requests.post(OVH_URL, headers=headers, json=payload, timeout=240)
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            return _normalize_image(_extract_image(response))
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(5 * attempt)
        except RuntimeError as exc:
            last_error = str(exc)
            status = getattr(locals().get("response"), "status_code", 0)
            if status not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                break
            time.sleep(5 * attempt)
    raise RuntimeError(last_error or "OVHcloud image generation failed")


def _route_image(prompt: str, refs: list[tuple[str, bytes, str]] | None = None) -> bytes:
    providers = (
        ("pollinations", lambda: _pollinations(prompt, refs)),
        ("ovhcloud", lambda: _ovhcloud(prompt)),
        ("cloudflare", lambda: _normalize_image(_CLOUDFLARE_ORIGINAL(prompt, refs))),
    )
    failures = []
    for provider, generate in providers:
        try:
            image = generate()
            if not image:
                raise RuntimeError("provider returned an empty image")
            PROVIDER_COUNTS[provider] += 1
            _record(provider, "success")
            return image
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            failures.append(f"{provider}: {detail}")
            _record(provider, "fallback", detail)
    raise RuntimeError("All configured image providers failed. " + " || ".join(failures))


# The base renderer calls this function for each storyboard beat. Keep its
# story prompts, reference loading, previous-frame continuity and OCR pipeline.
base._cloudflare = _route_image
_original_main = base.main


def main() -> None:
    PROVIDER_EVENTS.clear()
    for key in PROVIDER_COUNTS:
        PROVIDER_COUNTS[key] = 0
    _record("router", "started", "Pollinations -> OVHcloud -> Cloudflare")
    _original_main()

    manifest_path = base.RUN / "cloudflare_image_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["media_generation"] = "multi_provider_image_router"
        manifest["provider_priority"] = ["pollinations", "ovhcloud", "cloudflare"]
        manifest["provider_counts"] = PROVIDER_COUNTS
        manifest["provider_events"] = PROVIDER_EVENTS
        manifest["reference_policy"] = {
            "pollinations": "canonical references plus previous accepted frame via /v1/images/edits",
            "ovhcloud": "prompt-only SDXL; no direct reference-image input documented",
            "cloudflare": "canonical references plus previous accepted frame",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[PROVIDER SUMMARY] " + json.dumps(PROVIDER_COUNTS, sort_keys=True))


if __name__ == "__main__":
    main()
