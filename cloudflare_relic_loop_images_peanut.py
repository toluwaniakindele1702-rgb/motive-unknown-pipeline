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

# The two checked-in Chispa-Pifia images are the canonical visual bible.
base.CANONICAL_REFERENCE_PATHS = (
    ("Generate_modern_stickman_character_20261027121745.jpg", "image/jpeg"),
    ("Remove_bag_and_animate_image_20261027125028.jpg", "image/jpeg"),
)

base.STYLE_LOCK = """
CHISPA-PIFIA APPROVED 2D STICKMAN STYLE — HARD LOCK:
Use the supplied canonical reference images as the highest-priority visual authority.
The protagonist must remain the same simple, expressive 2D storytime stickman shown in
those references. Preserve the same large smooth pale round head, face construction,
eye shape and spacing, eyebrows, mouth style, head-to-body ratio, silhouette, limb
construction, outline thickness and overall proportions. Do not redesign the character.
Do not invent clothing or accessories that are not established by the references or story.

UNIFIED 2D ANIMATION WORLD:
The character, props, objects and every background element must all belong to one
coherent illustrated 2D cartoon world. Use clean hand-drawn/digital outlines, simplified
shapes, expressive poses, readable silhouettes, and restrained 2D cel-shading. Rooms,
furniture, streets, vehicles, trees, phones and all other objects must also look illustrated.
Never place the flat 2D stickman into a photograph, photorealistic environment or 3D scene.
Keep the established reference palette and rendering language consistent between shots.

CHARACTER AND SCENE CONTINUITY:
Reference 1 is the primary character identity anchor. Reference 2 is the secondary
approved style/action guide. Preserve the same character identity across every scene.
Expressions, poses, camera angles and story locations may change, but character design
must not. Use the previously accepted frame as an additional continuity aid without
overriding the canonical references or story facts.

ONE-SCENE RULE:
Generate exactly one full-frame vertical story shot per request. Never reproduce a
reference-sheet layout, collage, turnaround sheet or multiple panels.

ABSOLUTE EXCLUSIONS:
no photorealism, no live action, no 3D CGI, no anime/manga, no glossy mascot redesign,
no realistic human anatomy, no character redesign, no random outfit changes, no extra
limbs or fingers, no duplicated protagonist unless the story explicitly requires it,
no copied reference-sheet panels, no text, captions, speech bubbles, labels, logos
or watermarks.
""".strip()

base.NO_TEXT = """
STRICT ZERO-TEXT IMAGE RULE:
Do not render captions, subtitles, title cards, speech bubbles, labels, logos, watermarks,
UI, readable signs, letters, numbers, symbols, menus, posters, packaging text, screen text
or typography. Keep environmental surfaces blank or non-readable. Communicate the story
only through character acting, objects, composition, expressions and visible cause/effect.
""".strip()

base._reference_blurb = lambda: (
    "CANONICAL REFERENCE PRIORITY: Reference 1 defines the exact protagonist identity. "
    "Reference 2 defines the approved 2D visual language and action/style guidance. "
    "Match the original simple stickman construction, facial design, proportions, outlines "
    "and flat illustrated rendering. Do not copy the reference-sheet layout. The references "
    "define character appearance; the narration defines the action."
)

_original_scene_prompt = base._scene_prompt
def _peanut_scene_prompt(scene, characters, locations, previous_scene=None, attempt=1):
    cleaned = []
    for character in characters or []:
        item = dict(character) if isinstance(character, dict) else {}
        item["appearance"] = (
            "the approved canonical Chispa-Pifia 2D stickman from reference image 1; "
            "match its appearance exactly; do not redesign or invent wardrobe"
        )
        cleaned.append(item)
    return _original_scene_prompt(scene, cleaned, locations, previous_scene, attempt)

base._scene_prompt = _peanut_scene_prompt

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
