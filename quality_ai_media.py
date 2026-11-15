"""Direct image/audio/video-frame inspection adapters for Quality AI.

Uses the Google Gemini multimodal API when explicitly configured. Media stays
local until sent to the configured provider; no media is uploaded to GitHub or
YouTube. The module has no production-pipeline controls.
"""
from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

MAX_MEDIA_BYTES = 18 * 1024 * 1024
MAX_IMAGES = 8
IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
AUDIO_TYPES = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/aac", "audio/ogg", "audio/flac"}
VISION_PROMPT = """Inspect these actual storyboard/video frames for a short family-friendly
2D stickman comedy. Describe only visible evidence. Assess visual clarity,
character identity consistency, clothing/wardrobe continuity, location and
prop continuity, composition, and whether frames plausibly match the supplied
scene descriptions. Identify frame indexes for issues. Do not infer unseen
details. Return JSON: {"image_descriptions":[...],"observations":[...],
"continuity_issues":[...],"clarity_issues":[...],"uncertainties":[...]}."""
AUDIO_PROMPT = """Listen to this actual narration/audio. Assess intelligibility,
background noise, clipping/distortion, volume consistency, pacing, pauses,
pronunciation, and whether speech is complete. Do not infer visual content.
Return JSON: {"audio_observations":[...],"clarity_issues":[...],
"pacing_issues":[...],"technical_issues":[...],"uncertainties":[...]}."""


def build_gemini_client(environ: Mapping[str, str] | None = None) -> tuple[Any, str | None]:
    env = environ if environ is not None else os.environ
    key = (env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or "").strip()
    model = (env.get("QUALITY_AI_MULTIMODAL_MODEL") or "gemini-2.5-flash").strip()
    if not key:
        return None, None
    from google import genai
    return genai.Client(api_key=key), model


def _mime(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed in IMAGE_TYPES or guessed in AUDIO_TYPES:
        return guessed
    if path.suffix.lower() == ".mp3":
        return "audio/mpeg"
    if path.suffix.lower() in {".wav", ".wave"}:
        return "audio/wav"
    raise ValueError(f"Unsupported media type: {path.suffix or 'unknown'}")


def _read_media(path: str | Path, allowed: set[str]) -> tuple[bytes, str]:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"Media file not found: {file_path}")
    mime = _mime(file_path)
    if mime not in allowed:
        raise ValueError(f"Unsupported media MIME type: {mime}")
    size = file_path.stat().st_size
    if size <= 0 or size > MAX_MEDIA_BYTES:
        raise ValueError(f"Media file must be between 1 byte and {MAX_MEDIA_BYTES} bytes")
    return file_path.read_bytes(), mime


def _response_text(response: Any) -> str:
    value = getattr(response, "text", None)
    if not value:
        raise ValueError("Multimodal model returned an empty response")
    return str(value)


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Multimodal model did not return a JSON object")
    value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Multimodal response must be a JSON object")
    return value


def _generate(client: Any, model: str, parts: Sequence[Any], prompt: str) -> dict[str, Any]:
    from google.genai import types
    response = client.models.generate_content(
        model=model,
        contents=[*parts, prompt],
        config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
    )
    return _parse_json(_response_text(response))


def inspect_images(paths: Sequence[str | Path], client: Any, model: str) -> dict[str, Any]:
    if not paths:
        return {"image_descriptions": [], "observations": [], "continuity_issues": [],
                "clarity_issues": [], "uncertainties": ["No image files supplied."]}
    if len(paths) > MAX_IMAGES:
        raise ValueError(f"At most {MAX_IMAGES} images can be reviewed per request")
    from google.genai import types
    parts = []
    for index, path in enumerate(paths, start=1):
        data, mime = _read_media(path, IMAGE_TYPES)
        parts.extend([types.Part.from_text(text=f"Frame {index}:"), types.Part.from_bytes(data=data, mime_type=mime)])
    return _generate(client, model, parts, VISION_PROMPT)


def inspect_audio(path: str | Path, client: Any, model: str) -> dict[str, Any]:
    from google.genai import types
    data, mime = _read_media(path, AUDIO_TYPES)
    part = types.Part.from_bytes(data=data, mime_type=mime)
    return _generate(client, model, [part], AUDIO_PROMPT)


def extract_video_frames(video_path: str | Path, output_dir: str | Path,
                         max_frames: int = MAX_IMAGES) -> list[Path]:
    """Extract a bounded, evenly sampled set of frames with local ffmpeg."""
    if not 1 <= max_frames <= MAX_IMAGES:
        raise ValueError(f"max_frames must be between 1 and {MAX_IMAGES}")
    source = Path(video_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required for video inspection")
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    try:
        duration = float((probe.stdout or "").strip())
    except ValueError:
        duration = 0.0
    if probe.returncode != 0 or duration <= 0:
        raise RuntimeError("Could not determine video duration for even frame sampling")
    sample_rate = max_frames / duration
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    pattern = target / "quality-frame-%02d.jpg"
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vf", f"fps={sample_rate:.8f},scale=960:-2", "-frames:v", str(max_frames),
        "-q:v", "3", "-y", str(pattern),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    if completed.returncode != 0:
        raise RuntimeError("ffmpeg frame extraction failed: " + (completed.stderr or "")[-1000:])
    frames = sorted(target.glob("quality-frame-*.jpg"))[:max_frames]
    if not frames:
        raise RuntimeError("ffmpeg did not extract any video frames")
    return frames


def inspect_media_files(*, image_paths: Sequence[str | Path] = (),
                        audio_path: str | Path | None = None,
                        video_path: str | Path | None = None,
                        client: Any, model: str) -> dict[str, Any]:
    """Inspect actual media files and return evidence for review_submission."""
    frame_paths = list(image_paths)
    temp_context = None
    try:
        if video_path is not None:
            temp_context = tempfile.TemporaryDirectory(prefix="quality-ai-frames-")
            frame_paths.extend(extract_video_frames(video_path, temp_context.name))
        image_result = inspect_images(frame_paths, client, model) if frame_paths else {}
        extracted_audio = None
        if audio_path is None and video_path is not None:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("ffmpeg is required for audio extraction from video")
            extracted_audio = Path(temp_context.name) / "quality-audio.mp3"
            command = [
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
                "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", "-y", str(extracted_audio),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
            if completed.returncode != 0 or not extracted_audio.is_file() or extracted_audio.stat().st_size == 0:
                raise RuntimeError("ffmpeg audio extraction failed: " + (completed.stderr or "")[-1000:])
            audio_path = extracted_audio
        audio_result = inspect_audio(audio_path, client, model) if audio_path else {}
        return {
            "image_descriptions": image_result.get("image_descriptions", []),
            "visual_observations": image_result.get("observations", []),
            "continuity_issues": image_result.get("continuity_issues", []),
            "clarity_issues": image_result.get("clarity_issues", []),
            "audio_observations": audio_result.get("audio_observations", []),
            "audio_clarity_issues": audio_result.get("clarity_issues", []),
            "audio_pacing_issues": audio_result.get("pacing_issues", []),
            "audio_technical_issues": audio_result.get("technical_issues", []),
            "media_uncertainties": image_result.get("uncertainties", []) + audio_result.get("uncertainties", []),
            "media_review_provider": "google_gemini",
            "media_review_model": model,
            "media_files_reviewed": {"images_or_frames": len(frame_paths), "audio": bool(audio_path)},
        }
    finally:
        if temp_context is not None:
            temp_context.cleanup()



def review_media_submission(payload: Mapping[str, Any], *, client: Any, model: str,
                            image_paths: Sequence[str | Path] = (),
                            audio_path: str | Path | None = None,
                            video_path: str | Path | None = None,
                            text_client: Any = None,
                            text_model: str | None = None) -> dict[str, Any]:
    """End-to-end local media review adapter; still advisory and pipeline-agnostic."""
    from quality_ai import review_submission
    evidence = inspect_media_files(
        image_paths=image_paths, audio_path=audio_path, video_path=video_path,
        client=client, model=model,
    )
    enriched = dict(payload)
    enriched.update(evidence)
    review = review_submission(enriched, client=text_client, model=text_model)
    return {"review": review, "media_evidence": evidence}
