"""Relic Loop V2 media primitives: Nano Banana 2 image generation + Veo 3.1 animation.

This module is deliberately isolated from the production pipeline. It is used by the
2-minute V2 prototype first, then can be imported by the production scene planner.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from PIL import Image

try:
    from google import genai
    from google.genai import types
except Exception as exc:  # pragma: no cover
    genai = None
    types = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

# Use the current Gemini 3.1 Flash Image preview model for the generateContent
# image-to-video workflow documented by Google. The environment variable remains
# supported so the test/prod workflow can override it without editing code.
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview")
GEMINI_VIDEO_MODEL = os.environ.get("GEMINI_VIDEO_MODEL", "veo-3.1-generate-preview")


def client():
    if genai is None:
        raise RuntimeError(f"google-genai is unavailable: {_IMPORT_ERROR}")
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise RuntimeError("GEMINI_API_KEY is missing; V2 cannot generate media.")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip())


def generate_image(prompt: str, output_path: Path, reference_path: Optional[Path] = None) -> Path:
    """Generate a 16:9 scene image, optionally using a canonical RL reference."""
    c = client()
    contents = [prompt]
    if reference_path and reference_path.exists():
        contents.append(Image.open(reference_path).convert("RGB"))
        contents.append(
            "Use the supplied image as the canonical RL character reference. "
            "Preserve RL's face, hair, clothing, proportions and palette; change only "
            "pose, action, camera and environment requested by the scene prompt."
        )

    # The previous implementation passed response_format into
    # GenerateContentConfig. The installed google-genai SDK used by Actions rejects
    # that field, so use the SDK's ImageConfig for aspect ratio instead. Google also
    # documents this generateContent image workflow with response_modalities=['IMAGE'].
    response = c.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9"),
        ),
    )
    for part in response.parts:
        if getattr(part, "as_image", None):
            image = part.as_image()
            image.save(output_path)
            return output_path
    raise RuntimeError("Gemini image generation returned no image part.")


def generate_video_from_image(
    image_path: Path,
    prompt: str,
    output_path: Path,
    reference_path: Optional[Path] = None,
    timeout_seconds: int = 900,
) -> Path:
    """Animate a finished scene image with Veo 3.1.

    The scene image is the starting frame, so even when reference-image support is
    unavailable the generated motion is anchored to the exact still used in the edit.
    The optional RL reference is reserved for future reference-image API use.
    """
    del reference_path  # image-to-video is the reliable common denominator.
    c = client()
    base_image = Image.open(image_path).convert("RGB")
    operation = c.models.generate_videos(
        model=GEMINI_VIDEO_MODEL,
        prompt=prompt,
        image=base_image,
        config=types.GenerateVideosConfig(
            aspect_ratio="16:9",
            resolution="720p",
            person_generation="allow_adult",
        ),
    )
    started = time.time()
    while not operation.done:
        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"Veo generation timed out after {timeout_seconds}s")
        time.sleep(10)
        operation = c.operations.get(operation)
    generated = operation.response.generated_videos[0]
    c.files.download(file=generated.video)
    generated.video.save(output_path)
    if not output_path.exists() or output_path.stat().st_size < 10000:
        raise RuntimeError("Veo returned an invalid or empty video file.")
    return output_path


def animate_or_fallback(image_path: Path, prompt: str, video_path: Path, reference_path: Optional[Path] = None) -> bool:
    """Return True on animation success; never block the scene on a Veo failure."""
    try:
        generate_video_from_image(image_path, prompt, video_path, reference_path=reference_path)
        print(f"[V2 MOTION] animated: {video_path.name}")
        return True
    except Exception as exc:
        print(f"[V2 MOTION] animation failed; using image fallback for {image_path.name}: {exc}")
        return False
