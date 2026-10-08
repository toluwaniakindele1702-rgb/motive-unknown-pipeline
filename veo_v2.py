"""Relic Loop V2 media layer.

Cloudflare generates the stills. Gemini Omni Flash optionally animates selected
scene images. Animation is always best-effort: failures fall back to the still.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Iterable, Optional

import requests
from PIL import Image, ImageOps

try:
    from google import genai
except Exception as exc:  # pragma: no cover
    genai = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

CLOUDFLARE_IMAGE_MODEL = os.environ.get("CLOUDFLARE_IMAGE_MODEL", "@cf/black-forest-labs/flux-2-klein-4b")
GEMINI_VIDEO_MODEL = os.environ.get("GEMINI_VIDEO_MODEL", "gemini-omni-1.1-flash")
CLOUDFLARE_TIMEOUT = int(os.environ.get("CLOUDFLARE_IMAGE_TIMEOUT", "180"))


def _gemini_client():
    if genai is None:
        raise RuntimeError(f"google-genai is unavailable: {_IMPORT_ERROR}")
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing; video generation cannot run.")
    return genai.Client(api_key=key)


def _cloudflare_credentials() -> tuple[str, str]:
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not account or not token:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN are missing.")
    return account, token


def _reference_file(path: Path, size: int = 480) -> io.BytesIO:
    image = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (size, size), "white")
    contained = ImageOps.contain(image, (size - 20, size - 20), method=Image.Resampling.LANCZOS)
    canvas.paste(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _cloudflare_references(reference_paths: Iterable[Path]):
    refs = {}
    for index, path in enumerate(reference_paths):
        if path and path.exists():
            buf = _reference_file(path)
            refs[f"input_image_{index}"] = (f"rl_ref_{index}.png", buf.read(), "image/png")
    return refs


def _cloudflare_image(prompt: str, output_path: Path, references: list[Path]) -> Path:
    account, token = _cloudflare_credentials()
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{CLOUDFLARE_IMAGE_MODEL}"
    files = _cloudflare_references([p for p in references if p.exists()][:4])
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        data={"prompt": prompt, "width": "1024", "height": "576"},
        files=files or None,
        timeout=CLOUDFLARE_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success", True):
        raise RuntimeError(f"Cloudflare image generation failed: {payload}")
    result = payload.get("result") or {}
    encoded = result.get("image") if isinstance(result, dict) else None
    if not encoded:
        raise RuntimeError(f"Cloudflare returned no image data: {payload}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(encoded))
    if output_path.stat().st_size < 10000:
        raise RuntimeError("Cloudflare returned an invalid or empty image file.")
    return output_path


def generate_image(prompt: str, output_path: Path, reference_path: Optional[Path] = None,
                   reference_paths: Optional[list[Path]] = None) -> Path:
    refs = list(reference_paths or [])
    if reference_path and reference_path.exists() and reference_path not in refs:
        refs.insert(0, reference_path)
    return _cloudflare_image(prompt, output_path, refs[:4])


def _image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def generate_video_from_image(image_path: Path, prompt: str, output_path: Path,
                              reference_path: Optional[Path] = None,
                              reference_paths: Optional[list[Path]] = None) -> Path:
    """Use Gemini Omni image-to-video; optional RL images are explicit references."""
    client = _gemini_client()
    refs = list(reference_paths or [])
    if reference_path and reference_path.exists() and reference_path not in refs:
        refs.insert(0, reference_path)
    refs = [p for p in refs if p.exists()][:3]

    inputs = [{"type": "image", "data": _image_b64(image_path), "mime_type": "image/png"}]
    for ref in refs:
        inputs.append({"type": "image", "data": _image_b64(ref), "mime_type": "image/png"})

    if refs:
        role_text = "[# Sources <FIRST_FRAME>@Image1] [# References " + " ".join(
            f"<IMAGE_REF_{i}>@Image{i + 2}" for i in range(len(refs))
        ) + "]"
        task = "reference_to_video"
    else:
        role_text = "[# Sources <FIRST_FRAME>@Image1]"
        task = "image_to_video"

    inputs.append({
        "type": "text",
        "text": (
            role_text + "\nUse Image1 as the starting frame. "
            "Preserve the scene composition and animate only useful, natural movement. "
            "Keep RL's identity, face, hair, clothing, proportions and palette consistent. "
            "No subtitles, captions, logos, watermarks, or newly invented text.\n\n" + prompt
        ),
    })

    interaction = client.interactions.create(
        model=GEMINI_VIDEO_MODEL,
        input=inputs,
        generation_config={"video_config": {"task": task}},
    )
    output_video = getattr(interaction, "output_video", None)
    data = getattr(output_video, "data", None) if output_video else None
    if not data:
        raise RuntimeError(f"Gemini Omni returned no output video; status={getattr(interaction, 'status', None)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(data))
    if output_path.stat().st_size < 10000:
        raise RuntimeError("Gemini Omni returned an invalid or empty video file.")
    return output_path


def animate_or_fallback(image_path: Path, prompt: str, video_path: Path,
                         reference_path: Optional[Path] = None,
                         reference_paths: Optional[list[Path]] = None) -> bool:
    try:
        generate_video_from_image(image_path, prompt, video_path,
                                  reference_path=reference_path,
                                  reference_paths=reference_paths)
        print(f"[V2 MOTION] animated with {GEMINI_VIDEO_MODEL}: {video_path.name}")
        return True
    except Exception as exc:
        print(f"[V2 MOTION] animation failed; using image fallback for {image_path.name}: {exc}")
        return False
