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
from story_quality_gate import run_story_gate
from maintenance_ai import propose_repair
from relic_loop_media import (render_motion_video, build_sound_effect_track, build_background_music_track, write_scene_captions)
from quality_ai_media import inspect_audio, inspect_media_files, cloudflare_config

ROOT = Path(__file__).resolve().parent
OUT_ROOT = Path(os.getenv("RELIC_LOOP_OUTPUT_DIR", ROOT / "relic_loop_output")).resolve()
IMAGE_MODEL = os.getenv("CLOUDFLARE_IMAGE_MODEL", "@cf/black-forest-labs/flux-2-klein-4b")
WIDTH, HEIGHT = 576, 1024
MAX_SCENES = 12
MIN_SCENES = 10
HOOK_STYLES = ("curiosity gap", "in-medias-res action", "mystery reveal", "emotional contradiction", "unexpected reversal")

def _run(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd[:5])}; {(result.stderr or '')[-1800:]}")
    return result

def split_story_into_scenes(narration: str, count: int = 10) -> list[str]:
    """Split narration at sentence boundaries into balanced visual story beats."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", narration.strip()) if x.strip()]
    if len(sentences) < 5:
        raise ValueError("Narration needs at least five complete sentences for visual storyboarding")
    target_count = max(5, min(MAX_SCENES, count, len(sentences)))
    groups: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    remaining_words = sum(len(x.split()) for x in sentences)
    remaining_groups = target_count
    for sentence in sentences:
        sentence_words = len(sentence.split())
        target_words = max(1, remaining_words / max(1, remaining_groups))
        if current and len(groups) < target_count - 1 and current_words >= target_words:
            groups.append(current)
            remaining_groups -= 1
            current = []
            current_words = 0
            remaining_words -= sum(len(x.split()) for x in groups[-1])
        current.append(sentence)
        current_words += sentence_words
    if current:
        groups.append(current)
    beats = [" ".join(group) for group in groups if group]
    if len(beats) < 5:
        raise ValueError("Could not form enough meaningful sentence-level visual beats")
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

def _safe_moderation_retry_prompt(prompt: str, generic: bool = False) -> str:
    """Reframe a flagged beat as a neutral, family-friendly cartoon prompt."""
    def field(label: str, fallback: str) -> str:
        match = re.search(rf"^{re.escape(label)}:\s*(.+)$", prompt, re.M)
        return match.group(1).strip()[:900] if match else fallback
    character = field("CANONICAL CHARACTER LOCK", "one consistent expressive stickman in a charcoal hoodie and dark trousers")
    beat = ("the protagonist reacts with a surprised, playful expression to an everyday situation"
            if generic else field("EXACT NARRATION MOMENT",
                                  "the protagonist reacts to an amusing everyday misunderstanding"))
    setting = field("SETTING", "a simple everyday location")
    return (
        "Create one harmless, family-friendly 2D cartoon illustration in vertical 9:16. "
        f"Keep this exact character design consistent: {character}. "
        f"Show this lighthearted everyday comic moment: {beat}. "
        f"Location: {setting}. "
        "Use a clear expressive pose, readable face, clean bold outlines, coherent illustrated "
        "background and simple cinematic lighting. Keep the scene lighthearted and suitable "
        "for a broad audience. No text, captions, logos or watermarks. "
        "One frame only; preserve the same outfit and proportions."
    )

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
    try:
        blob = _cloudflare_image(prompt, refs)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "3030" not in message and "flagged" not in message:
            raise
        print("[IMAGE PROVIDER] Cloudflare flagged this prompt/reference combination; retrying a neutral reformulation without reference images")
        try:
            blob = _cloudflare_image(_safe_moderation_retry_prompt(prompt), None)
        except RuntimeError as retry_error:
            retry_message = str(retry_error).lower()
            if "3030" not in retry_message and "flagged" not in retry_message:
                raise
            print("[IMAGE PROVIDER] Beat-specific retry was also flagged; using a generic safe reaction frame for this beat")
            blob = _cloudflare_image(_safe_moderation_retry_prompt(prompt, generic=True), None)
    if not inspect_image(blob).valid:
        raise RuntimeError("Cloudflare image failed technical inspection")
    print("[IMAGE PROVIDER] Cloudflare generated an accepted image")
    return blob

async def _synthesize_voice(text: str, target: Path) -> str:
    # Keep the installed edge-tts package ahead of the repository's legacy
    # edge_tts.py module, which would otherwise shadow the package.
    import sys
    original_path = list(sys.path)
    sys.path[:] = [entry for entry in sys.path
                   if Path(entry or ".").resolve() != ROOT]
    try:
        import edge_tts
    finally:
        sys.path[:] = original_path

    preferred_voice = os.getenv("RELIC_LOOP_TTS_VOICE", "en-US-AndrewNeural").strip()
    rate = os.getenv("RELIC_LOOP_TTS_RATE", "+2%")
    pitch = os.getenv("RELIC_LOOP_TTS_PITCH", "+0Hz")
    voices = [preferred_voice]
    if preferred_voice != "en-US-GuyNeural":
        voices.append("en-US-GuyNeural")
    last_error = None
    for voice in voices:
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await communicate.save(str(target))
            if not target.is_file() or target.stat().st_size < 5000:
                raise RuntimeError("Edge TTS did not produce a usable narration track")
            print(f"[VOICE] narration generated with {voice} at {rate}")
            return voice
        except Exception as exc:
            last_error = exc
            if target.exists():
                target.unlink()
            print(f"[VOICE] {voice} unavailable; trying the fallback voice: {type(exc).__name__}")
    raise RuntimeError(f"All configured Edge TTS voices failed: {last_error}")

def _duration(path: Path) -> float:
    raw = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path)], 30).stdout.strip()
    value = float(raw)
    if value <= 0: raise RuntimeError("Media duration is invalid")
    return value

def _assemble(images: list[Path], beats: list[str], narration_audio: Path, output: Path):
    """Render moving scenes, synchronized captions, original music and story-aware SFX."""
    duration = _duration(narration_audio)
    counts = [max(1, len(x.split())) for x in beats]
    total = sum(counts)
    durations = [duration * n / total for n in counts]
    motion_video = output.parent / "motion_video.mp4"
    motion_report = render_motion_video(images, durations, motion_video, WIDTH, HEIGHT, fps=30)
    sfx_path = output.parent / "story_sound_effects.wav"
    sfx_report = build_sound_effect_track(beats, durations, sfx_path)
    music_path = output.parent / "background_music.wav"
    joined = " ".join(beats).lower()
    mood = ("mysterious" if any(x in joined for x in ("secret", "mystery", "shadow", "strange"))
            else "emotional" if any(x in joined for x in ("lonely", "missed", "heart", "sad"))
            else "warm" if any(x in joined for x in ("family", "friend", "thank", "kind"))
            else "playful")
    music_report = build_background_music_track(durations, music_path, mood=mood)
    captions_path = output.parent / "captions.srt"
    caption_report = write_scene_captions(beats, durations, captions_path)
    subtitle_path = str(captions_path.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    _run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(motion_video), "-i", str(narration_audio), "-i", str(sfx_path), "-i", str(music_path),
        "-filter_complex",
        "[1:a]aresample=48000,volume=1.0[narration];"
        "[2:a]aresample=48000,volume=0.34[sfx];"
        "[3:a]aresample=48000,volume=0.22[music];"
        "[narration][sfx][music]amix=inputs=3:duration=first:dropout_transition=0,"
        "alimiter=limit=0.92[audio]",
        "-vf", f"subtitles='{subtitle_path}':force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00101010,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=95'",
        "-map", "0:v:0", "-map", "[audio]", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-t", f"{duration:.3f}", "-movflags", "+faststart", str(output),
    ], timeout=360)
    if not output.is_file() or output.stat().st_size < 100_000:
        raise RuntimeError("FFmpeg did not produce a usable captioned MP4 with narration, music and sound effects")
    return _duration(output), motion_report, sfx_report, music_report, caption_report

def _generate_valid_story(client, model: str, topic: str, feedback: QualityFeedback,
                         recent_story_summaries: list[str] | None = None) -> dict:
    """Retry out-of-range drafts with progressively tighter word-count instructions."""
    last_error = None
    hook_style = HOOK_STYLES[(int(time.strftime("%j")) + len(topic or "")) % len(HOOK_STYLES)]
    for target_words in ((180, 215), (180, 205), (180, 195)):
        brief = CreativeBrief(topic=topic, hook_style=hook_style,
                              duration_seconds=(50, 90), target_words=target_words)
        try:
            story = generate_story(client, model, brief, feedback, recent_story_summaries or [])
            story["hook_style"] = hook_style
            return story
        except ValueError as exc:
            if "words; expected" not in str(exc):
                raise
            last_error = exc
            print(f"[CREATIVE AI] Draft outside {target_words[0]}-{target_words[1]} words; requesting a tighter rewrite")
    raise ValueError(f"Creative AI could not produce a story within 180-215 words after three attempts: {last_error}")


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
    if previous_feedback is None:
        feedback_path = OUT_ROOT / "latest_quality_feedback.json"
        if feedback_path.is_file():
            try:
                previous_feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
                print("[QUALITY FEEDBACK] Loaded advisory feedback from the previous run")
            except (OSError, json.JSONDecodeError):
                previous_feedback = None
    feedback = QualityFeedback.from_mapping(previous_feedback)
    creative_model = (os.getenv("CREATIVE_AI_MODEL", "").strip() or "openai/gpt-oss-120b")
    quality_model = (os.getenv("QUALITY_AI_MODEL", "").strip() or creative_model)
    def draft_story(coaching):
        return _generate_valid_story(client, creative_model, topic,
            QualityFeedback.from_mapping(coaching), recent_story_summaries)
    story, story_reviews, coached_feedback = run_story_gate(
        client, quality_model, draft_story, feedback, max_rewrites=2
    )
    report["story"] = story
    report["story_quality_reviews"] = story_reviews
    report["story_quality_summary"] = {
        "candidate_count": sum(1 for r in story_reviews if "candidate" in r),
        "attempts": len(story_reviews),
        "review_count": len(story_reviews),
        "last_overall_score": next((r.get("overall_score") for r in reversed(story_reviews) if r.get("overall_score") is not None), None),
        "rewrite_attempts": sum(1 for r in story_reviews if "rewrite_attempt" in r),
        "selected_hook_style": story.get("hook_style"),
        "advisory_only": True, "production_blocked": False,
    }
    report["stages"].append({"name": "creative_ai", "status": "completed"})
    report["stages"].append({"name": "story_quality_coaching", "status": "completed", "attempts": len(story_reviews), "advisory_only": True})
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
    voice_used = asyncio.run(_synthesize_voice(story["narration"], audio_path))
    report["stages"].append({"name": "voice_generation", "status": "completed", "voice": voice_used, "rate": os.getenv("RELIC_LOOP_TTS_RATE", "+2%")})
    video_path = run_dir / "relic_loop_video.mp4"
    actual_duration, motion_report, sfx_report, music_report, caption_report = _assemble(images, beats, audio_path, video_path)
    report["camera_motion"] = motion_report
    report["sound_effects"] = sfx_report
    report["background_music"] = music_report
    report["captions"] = caption_report
    report["video"] = {"path": str(video_path), "duration_seconds": round(actual_duration, 2),
                       "width": WIDTH, "height": HEIGHT, "fps": 30, "codec": "h264+aac"}
    report["stages"].append({"name": "video_assembly", "status": "completed"})

    # Quality AI always runs. Media vision is attempted when Cloudflare credentials
    # are configured; local audio diagnostics and deterministic checks remain available.
    evidence: dict[str, Any] = {"title": story["title"], "narration": story["narration"],
                                "duration_seconds": actual_duration, "scenes": beats}
    try:
        account, token, vision_model = cloudflare_config()
        media = inspect_media_files(image_paths=[], audio_path=None,
                                    video_path=str(video_path), account_id=account,
                                    api_token=token, model=vision_model, scene_descriptions=beats)
        evidence.update(media)
        report["stages"].append({"name": "quality_media_inspection", "status": "completed"})
    except Exception as exc:
        try:
            # Even if cloud vision is unavailable, inspect the actual final mix,
            # not merely the narration source track.
            with tempfile.TemporaryDirectory(prefix="relic-loop-final-audio-") as temp_dir:
                final_audio = Path(temp_dir) / "final_mix.wav"
                _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                      "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
                      str(final_audio)], timeout=90)
                evidence.update(inspect_audio(str(final_audio)))
        except Exception as audio_exc:
            evidence["technical_observations"] = [f"Final-mix audio diagnostics unavailable: {audio_exc}"]
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
                source_names = ("creative_ai.py", "creative_ai_image_generator.py", "quality_ai.py",
                                "quality_ai_media.py", "maintenance_ai.py", "relic_loop_video.py")
                source_files = {name: (ROOT / name).read_text(encoding="utf-8")[:3500]
                                for name in source_names if (ROOT / name).is_file()}
                repair = propose_repair(failure["error"], source_files, Groq(api_key=key))
                repair.pop("patch", None)
                failure["maintenance_ai"] = repair
        except Exception as diag:
            failure["maintenance_ai"] = {"diagnostic_error": str(diag)[:500], "deployment": "not_performed"}
        print(json.dumps(failure, indent=2))
        raise
