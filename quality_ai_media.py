"""Cloudflare Workers AI vision + local FFmpeg audio diagnostics for Quality AI."""
from __future__ import annotations
import base64, json, mimetypes, os, re, shutil, subprocess, tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
import requests

MAX_IMAGES = 8
MAX_MEDIA_BYTES = 18 * 1024 * 1024
DEFAULT_VISION_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct"
IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
AUDIO_TYPES = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/aac", "audio/ogg", "audio/flac"}

def cloudflare_config(environ: Mapping[str, str] | None = None):
    env = environ if environ is not None else os.environ
    return ((env.get("CLOUDFLARE_ACCOUNT_ID") or "").strip() or None,
            (env.get("CLOUDFLARE_API_TOKEN") or env.get("CLOUDFLARE_AUTH_TOKEN") or "").strip() or None,
            (env.get("QUALITY_AI_VISION_MODEL") or DEFAULT_VISION_MODEL).strip())

def _mime(path: Path):
    value = mimetypes.guess_type(path.name)[0]
    if value in IMAGE_TYPES | AUDIO_TYPES: return value
    if path.suffix.lower() == ".mp3": return "audio/mpeg"
    if path.suffix.lower() in {".wav", ".wave"}: return "audio/wav"
    raise ValueError(f"Unsupported media type: {path.suffix or 'unknown'}")

def _read(path, allowed):
    p = Path(path).expanduser()
    if not p.is_file(): raise FileNotFoundError(f"Media file not found: {p}")
    mime = _mime(p)
    if mime not in allowed: raise ValueError(f"Unsupported media MIME type: {mime}")
    size = p.stat().st_size
    if size <= 0 or size > MAX_MEDIA_BYTES: raise ValueError(f"Media file must be between 1 byte and {MAX_MEDIA_BYTES} bytes")
    return p.read_bytes(), mime

def _parse_json(raw):
    value = (raw or "").strip()
    if value.startswith("```"): value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    a, b = value.find("{"), value.rfind("}")
    if a < 0 or b < a: raise ValueError("Cloudflare vision model did not return a JSON object")
    result = json.loads(value[a:b+1])
    if not isinstance(result, dict): raise ValueError("Vision response must be a JSON object")
    return result

def _vision(data, prompt, account_id, api_token, model, session=requests):
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    response = session.post(url, headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
                            json={"prompt": prompt, "image": base64.b64encode(data).decode("ascii")}, timeout=90)
    response.raise_for_status()
    payload = response.json()
    result = payload.get("result", payload) if isinstance(payload, dict) else payload
    if isinstance(result, dict):
        text = result.get("response") or result.get("description") or result.get("result")
        if isinstance(text, dict): return text
        if isinstance(text, str): return _parse_json(text)
        if "image_description" in result: return result
    if isinstance(result, str): return _parse_json(result)
    raise ValueError("Cloudflare vision response had an unsupported shape")

def inspect_images(paths: Sequence[str | Path], *, account_id, api_token, model=DEFAULT_VISION_MODEL,
                   scene_descriptions=(), session=requests):
    if len(paths) > MAX_IMAGES: raise ValueError(f"At most {MAX_IMAGES} images can be reviewed per request")
    out = {"image_descriptions": [], "observations": [], "continuity_issues": [], "clarity_issues": [], "uncertainties": []}
    for i, path in enumerate(paths, 1):
        data, _ = _read(path, IMAGE_TYPES)
        scene = scene_descriptions[i-1] if i <= len(scene_descriptions) else ""
        prompt = ("Inspect this actual frame from a short family-friendly 2D stickman comedy. Describe only visible evidence. "
                  "Assess clarity, character identity, wardrobe, location/prop continuity, composition and match to scene. "
                  "Do not infer unseen details. Return JSON keys image_description, observations, continuity_issues, clarity_issues, uncertainties. "
                  f"Frame {i}. Expected scene: {scene[:1200]}")
        item = _vision(data, prompt, account_id, api_token, model, session)
        out["image_descriptions"].append(str(item.get("image_description", ""))[:1000])
        for key in ("observations", "continuity_issues", "clarity_issues", "uncertainties"):
            value = item.get(key, [])
            if isinstance(value, str): value = [value]
            if isinstance(value, list): out[key].extend(f"Frame {i}: {str(x)[:500]}" for x in value[:6])
    return out

def inspect_audio(path):
    _read(path, AUDIO_TYPES)
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe: raise RuntimeError("ffmpeg and ffprobe are required for local audio diagnostics")
    src = str(Path(path).expanduser())
    probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", src],
                           capture_output=True, text=True, timeout=30, check=False)
    if probe.returncode: raise RuntimeError("Could not read audio duration")
    duration = float(probe.stdout.strip())
    measured = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-i", src, "-af",
                                "volumedetect,silencedetect=noise=-45dB:d=0.35", "-f", "null", "-"],
                               capture_output=True, text=True, timeout=90, check=False)
    log = measured.stderr or ""
    mean = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", log)
    peak = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", log)
    silences = len(re.findall(r"silence_start:", log))
    mean_db = float(mean.group(1)) if mean else None
    peak_db = float(peak.group(1)) if peak else None
    issues, recs = [], []
    if peak_db is not None and peak_db >= -0.1:
        issues.append("Peak level is at or very near 0 dBFS; clipping risk should be checked.")
        recs.append("Reduce peak level and recheck for clipping.")
    if mean_db is not None and mean_db < -30:
        issues.append("Mean audio level is very quiet.")
        recs.append("Raise narration loudness while preserving headroom.")
    if silences: issues.append(f"{silences} silence interval(s) longer than 350 ms detected.")
    return {"audio_observations": [f"Duration: {duration:.2f}s.",
            f"Mean level: {mean_db:.1f} dBFS." if mean_db is not None else "Mean level unavailable.",
            f"Peak level: {peak_db:.1f} dBFS." if peak_db is not None else "Peak level unavailable.",
            f"Silence intervals: {silences}."],
            "technical_issues": issues, "recommendations": recs,
            "not_assessed": ["Speech intelligibility, pronunciation, delivery and semantic completeness are not assessed by signal metrics."],
            "duration_seconds": duration}

def extract_video_frames(video_path, output_dir, max_frames=MAX_IMAGES):
    if not 1 <= max_frames <= MAX_IMAGES: raise ValueError("max_frames is outside supported range")
    source = Path(video_path).expanduser()
    if not source.is_file(): raise FileNotFoundError(f"Video file not found: {source}")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe: raise RuntimeError("ffmpeg and ffprobe are required for video inspection")
    probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", str(source)], capture_output=True, text=True, timeout=30)
    duration = float(probe.stdout.strip()) if probe.returncode == 0 else 0
    if duration <= 0: raise RuntimeError("Could not determine video duration")
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("quality-frame-*.jpg"): old.unlink()
    pattern = target / "quality-frame-%02d.jpg"
    result = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(source),
                             "-vf", f"fps={max_frames/duration:.8f},scale=960:-2", "-frames:v", str(max_frames),
                             "-q:v", "3", "-y", str(pattern)], capture_output=True, text=True, timeout=90)
    if result.returncode: raise RuntimeError("ffmpeg frame extraction failed: " + (result.stderr or "")[-1000:])
    frames = sorted(target.glob("quality-frame-*.jpg"))[:max_frames]
    if not frames: raise RuntimeError("ffmpeg did not extract any video frames")
    return frames

def inspect_media_files(*, image_paths=(), audio_path=None, video_path=None, account_id=None, api_token=None,
                        model=DEFAULT_VISION_MODEL, scene_descriptions=(), session=requests):
    frames = list(image_paths); temp = None
    try:
        if video_path:
            temp = tempfile.TemporaryDirectory(prefix="quality-ai-frames-")
            frames.extend(extract_video_frames(video_path, temp.name))
        if frames and (not account_id or not api_token):
            raise RuntimeError("Cloudflare vision requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (or CLOUDFLARE_AUTH_TOKEN).")
        visual = inspect_images(frames, account_id=account_id, api_token=api_token, model=model,
                                 scene_descriptions=scene_descriptions, session=session) if frames else {}
        if audio_path is None and video_path:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg: raise RuntimeError("ffmpeg is required for audio extraction")
            audio_path = Path(temp.name) / "quality-audio.mp3"
            result = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
                                     "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", "-y", str(audio_path)],
                                    capture_output=True, text=True, timeout=90)
            if result.returncode or not audio_path.is_file() or not audio_path.stat().st_size:
                raise RuntimeError("ffmpeg audio extraction failed: " + (result.stderr or "")[-1000:])
        audio = inspect_audio(audio_path) if audio_path else {}
        return {"image_descriptions": visual.get("image_descriptions", []),
                "visual_observations": visual.get("observations", []),
                "continuity_issues": visual.get("continuity_issues", []),
                "clarity_issues": visual.get("clarity_issues", []),
                "audio_observations": audio.get("audio_observations", []),
                "audio_technical_issues": audio.get("technical_issues", []),
                "audio_recommendations": audio.get("recommendations", []),
                "audio_not_assessed": audio.get("not_assessed", []),
                "media_uncertainties": visual.get("uncertainties", []),
                "media_review_provider": "cloudflare_workers_ai" if frames else "local_ffmpeg_only",
                "media_review_model": model if frames else None,
                "media_files_reviewed": {"images_or_frames": len(frames), "audio": bool(audio_path)}}
    finally:
        if temp: temp.cleanup()

def review_media_submission(payload: Mapping[str, Any], *, image_paths=(), audio_path=None, video_path=None,
                            account_id=None, api_token=None, model=None, session=requests):
    from quality_ai import review_submission
    env_account, env_token, env_model = cloudflare_config()
    evidence = inspect_media_files(image_paths=image_paths, audio_path=audio_path, video_path=video_path,
                                   account_id=account_id or env_account, api_token=api_token or env_token,
                                   model=model or env_model, session=session)
    enriched = dict(payload); enriched.update(evidence)
    return {"review": review_submission(enriched), "media_evidence": evidence}
