#!/usr/bin/env python3
"""Independent Relic Loop video producer: Creative AI -> images -> voice -> MP4 -> QA.

No YouTube API, OAuth, publishing, or legacy production imports are used here.
Raphael is attempted when a session is configured; Cloudflare FLUX.2 is fallback.
"""
from __future__ import annotations
import asyncio, json, os, re, shutil, subprocess, tempfile, time
from pathlib import Path
from typing import Any
import requests
from PIL import Image
from io import BytesIO

from creative_ai import CreativeBrief, QualityFeedback, generate_story, create_visual_brief, generate_scene_image
from creative_ai_image_generator import inspect_image
from quality_ai import build_client_from_env, review_submission
from maintenance_ai import propose_repair
from quality_ai_media import inspect_audio, inspect_media_files, cloudflare_config

ROOT = Path(__file__).resolve().parent
OUT_ROOT = Path(os.getenv("RELIC_LOOP_OUTPUT_DIR", ROOT / "relic_loop_output")).resolve()
IMAGE_MODEL = os.getenv("CLOUDFLARE_IMAGE_MODEL", "@cf/black-forest-labs/flux-2-klein-4b")
WIDTH, HEIGHT = 576, 1024
MAX_SCENES = 12
MIN_SCENES = 10

def _run(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd[:5])}; {(result.stderr or '')[-1800:]}")
    return result

def split_story_into_scenes(narration: str, count: int = 10) -> list[str]:
    """Group complete sentences into 10-12 visual beats without changing narration."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", narration.strip()) if x.strip()]
    if len(sentences) < 5:
        raise ValueError("Narration has too few complete sentences for a visual storyboard")
    count = max(MIN_SCENES, min(MAX_SCENES, count, len(sentences)))
    groups: list[list[str]] = [[] for _ in range(count)]
    # Greedy word balancing while preserving sentence order.
    totals = [0] * count
    for sentence in sentences:
        target = min(range(count), key=lambda i: (totals[i], i))
        groups[target].append(sentence)
        totals[target] += len(sentence.split())
    # Empty groups can occur with short sentence counts; merge down and enforce >=5.
    beats = [" ".join(group) for group in groups if group]
    while len(beats) < MIN_SCENES and len(sentences) > len(beats):
        break
    if len(beats) < 5:
        raise ValueError("Could not form enough meaningful visual beats")
    return beats

def _cloudflare_image(prompt: str, refs: list[tuple[str, bytes, str]] | None = None) -> bytes:
    account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = (os.getenv("CLOUDFLARE_API_TOKEN") or os.getenv("CLOUDFLARE_AUTH_TOKEN") or "").strip()
    if not account or not token:
        raise RuntimeError("Cloudflare image fallback requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN")
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{IMAGE_MODEL}"
    form: dict[str, Any] = {"prompt": (None, prompt), "width": (None, str(WIDTH)), "height": (None, str(HEIGHT))}
    for i, (name, blob, mime) in enumerate((refs or [])[:3]):
        with Image.open(BytesIO(blob)) as im:
            im = im.convert("RGB"); im.thumbnail((511, 511), Image.Resampling.LANCZOS)
            buf = BytesIO(); im.save(buf, format="JPEG", quality=88)
        form[f"input_image_{i}"] = (f"reference_{i}.jpg", buf.getvalue(), "image/jpeg")
    response = requests.post(url, headers={"Authorization": f"Bearer {token}"}, files=form, timeout=180)
    if not response.ok:
        raise RuntimeError(f"Cloudflare image API HTTP {response.status_code}: {response.text[:1000]}")
    content_type = response.headers.get("content-type", "")
    if "image/" in content_type:
        blob = response.content
    else:
        payload = response.json()
        result = payload.get("result") or {}
        blob = result.get("image")
        if isinstance(blob, str):
            import base64
            blob = base64.b64decode(blob)
        if not isinstance(blob, bytes):
            raise RuntimeError("Cloudflare returned no image bytes")
    if len(blob) < 2048:
        raise RuntimeError("Cloudflare returned an empty or undersized image")
    return blob

def image_provider(prompt: str, refs=None) -> bytes:
    """Prefer Raphael only when a saved session exists; otherwise use Cloudflare."""
    raphael_enabled = os.getenv("RELIC_LOOP_USE_RAPHAEL", "1").lower() not in {"0", "false", "no"}
    session_path = os.getenv("RAPHAEL_STORAGE_STATE", "").strip()
    session_json = os.getenv("RAPHAEL_STORAGE_STATE_JSON", "").strip()
    if raphael_enabled and (session_json or (session_path and Path(session_path).is_file())):
        try:
            from raphael_browser import generate as raphael_generate
            blob = raphael_generate(prompt, refs)
            if inspect_image(blob).valid:
                print("[IMAGE PROVIDER] Raphael generated an accepted image")
                return blob
            raise RuntimeError("Raphael image failed technical inspection")
        except Exception as exc:
            print(f"[IMAGE PROVIDER] Raphael unavailable; falling back to Cloudflare: {type(exc).__name__}: {exc}")
    else:
        print("[IMAGE PROVIDER] Raphael session not configured; using Cloudflare fallback")
    blob = _cloudflare_image(prompt, refs)
    if not inspect_image(blob).valid:
        raise RuntimeError("Cloudflare image failed technical inspection")
    print("[IMAGE PROVIDER] Cloudflare generated an accepted image")
    return blob

async def _synthesize_voice(text: str, target: Path) -> None:
    import edge_tts
    voice = os.getenv("RELIC_LOOP_TTS_VOICE", "en-US-GuyNeural")
    rate = os.getenv("RELIC_LOOP_TTS_RATE", "+0%")
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(str(target))
    if not target.is_file() or target.stat().st_size < 5000:
        raise RuntimeError("Edge TTS did not produce a usable narration track")

def _duration(path: Path) -> float:
    raw = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path)], 30).stdout.strip()
    value = float(raw)
    if value <= 0: raise RuntimeError("Media duration is invalid")
    return value

def _assemble(images: list[Path], beats: list[str], narration_audio: Path, output: Path) -> float:
    duration = _duration(narration_audio)
    counts = [max(1, len(x.split())) for x in beats]
    total = sum(counts)
    durations = [duration * n / total for n in counts]
    concat = output.parent / "images.concat.txt"
    lines = []
    for image, seconds in zip(images, durations):
        escaped = str(image.resolve()).replace("'", "'\\''")
        lines += [f"file '{escaped}'", f"duration {seconds:.6f}"]
    last = str(images[-1].resolve()).replace("'", "'\\''")
    lines.append(f"file '{last}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
          "-f", "concat", "-safe", "0", "-i", str(concat), "-i", str(narration_audio),
          "-map", "0:v:0", "-map", "1:a:0", "-vf",
          f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1,fps=30,format=yuv420p",
          "-c:v", "libx264", "-preset", "medium", "-crf", "21",
          "-c:a", "aac", "-b:a", "160k", "-t", f"{duration:.3f}",
          "-movflags", "+faststart", str(output)], timeout=300)
    if not output.is_file() or output.stat().st_size < 100_000:
        raise RuntimeError("FFmpeg did not produce a usable MP4")
    return _duration(output)

def produce_video(topic: str = "", previous_feedback: dict | None = None,
                  recent_story_summaries: list[str] | None = None) -> dict:
    """Generate a complete local MP4, review it, and save an auditable run package."""
    started = time.time()
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_dir = OUT_ROOT / run_id
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "pipeline": "relic-loop-independent-video-v1", "run_id": run_id,
        "status": "running", "production_touched": False, "published": False,
        "youtube_auth_required": False, "stages": [],
    }
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is missing; cannot generate the story")
    from groq import Groq
    client = Groq(api_key=groq_key)
    feedback = QualityFeedback.from_mapping(previous_feedback)
    story = generate_story(client, os.getenv("CREATIVE_AI_MODEL", "openai/gpt-oss-120b"),
                           CreativeBrief(topic=topic, duration_seconds=(50, 90), target_words=(180, 215)),
                           feedback, recent_story_summaries or [])
    report["story"] = story
    report["stages"].append({"name": "creative_ai", "status": "completed"})
    beats = split_story_into_scenes(story["narration"], count=10)
    canonical = ("one consistent expressive 2D stickman with a large pale round head, bold oval eyes, black eyebrows, simple mouth, charcoal hoodie, dark cargo trousers and black-and-white high-top sneakers; same exact model and outfit in every frame")
    images: list[Path] = []
    previous_image: bytes | None = None
    for index, beat in enumerate(beats, 1):
        brief = create_visual_brief(
            index, beat, canonical, story["setting"],
            previous_scene_state=("Continue directly from the prior accepted frame. Preserve outfit, prop state, room layout, lighting direction and screen direction." if index > 1 else ""),
            shot_direction="vertical 9:16 cinematic 2D cartoon action shot; vary wide, medium and close-up framing to match the beat",
        )
        refs = [("previous_accepted_scene.png", previous_image, "image/png")] if previous_image else None
        path = image_dir / f"scene_{index:02d}.png"
        blob, meta = generate_scene_image(brief, image_provider, references=refs, output_path=path)
        previous_image = blob
        images.append(path)
        report.setdefault("image_generation", []).append({
            "scene": index, "provider_attempts": meta["attempts"],
            "dimensions": meta["final_inspection"],
        })
        print(f"[VIDEO] image {index}/{len(beats)} complete")
    report["stages"].append({"name": "image_generation", "status": "completed", "count": len(images)})

    audio_path = run_dir / "narration.mp3"
    asyncio.run(_synthesize_voice(story["narration"], audio_path))
    report["stages"].append({"name": "voice_generation", "status": "completed", "voice": os.getenv("RELIC_LOOP_TTS_VOICE", "en-US-GuyNeural")})
    video_path = run_dir / "relic_loop_video.mp4"
    actual_duration = _assemble(images, beats, audio_path, video_path)
    report["video"] = {"path": str(video_path), "duration_seconds": round(actual_duration, 2),
                       "width": WIDTH, "height": HEIGHT, "fps": 30, "codec": "h264+aac"}
    report["stages"].append({"name": "video_assembly", "status": "completed"})

    # Quality AI always runs. Media vision is attempted when Cloudflare credentials
    # are configured; local audio diagnostics and deterministic checks remain available.
    evidence: dict[str, Any] = {"title": story["title"], "narration": story["narration"],
                                "duration_seconds": actual_duration, "scenes": beats}
    try:
        account, token, vision_model = cloudflare_config()
        media = inspect_media_files(image_paths=images, audio_path=str(audio_path),
                                    video_path=str(video_path), account_id=account,
                                    api_token=token, model=vision_model, scene_descriptions=beats)
        evidence.update(media)
        report["stages"].append({"name": "quality_media_inspection", "status": "completed"})
    except Exception as exc:
        try:
            evidence.update(inspect_audio(str(audio_path)))
        except Exception as audio_exc:
            evidence["technical_observations"] = [f"Audio diagnostics unavailable: {audio_exc}"]
        evidence["media_review_unavailable"] = f"{type(exc).__name__}: {exc}"
        report["stages"].append({"name": "quality_media_inspection", "status": "partial",
                                 "reason": str(exc)[:500]})
    qclient, qmodel = build_client_from_env()
    review = review_submission(evidence, client=qclient, model=qmodel)
    report["quality_review"] = review
    report["next_creative_feedback"] = review["creative_ai_feedback"]
    report["stages"].append({"name": "quality_ai", "status": "completed", "advisory_only": True})

    # Save feedback for the next run in the same workspace; it is not a production gate.
    (OUT_ROOT / "latest_quality_feedback.json").write_text(
        json.dumps(report["next_creative_feedback"], indent=2, ensure_ascii=False), encoding="utf-8")
    report["status"] = "completed"
    report["elapsed_seconds"] = round(time.time() - started, 1)
    report["artifacts"] = {"video": str(video_path), "audio": str(audio_path),
                           "images": [str(x) for x in images]}
    (run_dir / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "title": story["title"],
                      "duration_seconds": actual_duration, "video": str(video_path),
                      "quality_score": review.get("overall_score")}, indent=2))
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="")
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()
    if args.output_root:
        OUT_ROOT = Path(args.output_root).resolve()
    try:
        result = produce_video(args.topic)
    except Exception as exc:
        failure = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "production_touched": False, "published": False, "deployment": "not_performed"}
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "failure_report.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        try:
            key = os.getenv("GROQ_API_KEY", "").strip()
            if key:
                from groq import Groq
                repair = propose_repair(failure["error"], {}, Groq(api_key=key))
                repair.pop("patch", None)
                failure["maintenance_ai"] = repair
        except Exception as diag:
            failure["maintenance_ai"] = {"diagnostic_error": str(diag)[:500], "deployment": "not_performed"}
        print(json.dumps(failure, indent=2))
        raise
