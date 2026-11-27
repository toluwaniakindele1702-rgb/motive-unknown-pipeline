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
from maintenance_ai import propose_repair, validate_patch_in_sandbox as maintenance_ai_validate
from relic_loop_media import (render_motion_video, build_sound_effect_track, build_background_music_track, write_word_level_captions)
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

async def _synthesize_voice(text: str, target: Path) -> tuple[str, list[dict]]:
    """Create a performed voice track: narrator for prose, alternating character
    voices for quoted dialogue, with word timings kept on the combined timeline."""
    import sys
    original_path = list(sys.path)
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT]
    try:
        import edge_tts
    finally:
        sys.path[:] = original_path

    preferred_voice = os.getenv("RELIC_LOOP_TTS_VOICE", "en-US-AndrewNeural").strip()
    rate = os.getenv("RELIC_LOOP_TTS_RATE", "+0%")
    pitch = os.getenv("RELIC_LOOP_TTS_PITCH", "+0Hz")
    dialogue_voices = [
        os.getenv("RELIC_LOOP_DIALOGUE_VOICE_A", "en-US-JennyNeural").strip(),
        os.getenv("RELIC_LOOP_DIALOGUE_VOICE_B", "en-US-GuyNeural").strip(),
    ]

    # Preserve every character and punctuation mark. Quoted dialogue is performed
    # separately so it no longer gets flattened into the narrator's single read.
    pieces = re.split(r'(["“][^"”]+["”])', text)
    segments: list[tuple[str, str]] = []
    dialogue_index = 0
    for piece in pieces:
        clean = piece.strip()
        if not clean:
            continue
        quoted = ((clean.startswith('"') and clean.endswith('"')) or
                  (clean.startswith('“') and clean.endswith('”')))
        spoken = clean[1:-1].strip() if quoted else clean
        if spoken:
            voice = dialogue_voices[dialogue_index % len(dialogue_voices)] if quoted else preferred_voice
            if quoted:
                dialogue_index += 1
            segments.append((spoken, voice))
    if not segments:
        raise ValueError("Voice synthesis received no speakable text")

    word_cues: list[dict] = []
    voice_sequence: list[str] = []
    cursor = 0.0
    with tempfile.TemporaryDirectory(prefix="relic-loop-voice-") as folder:
        root = Path(folder)
        concat_lines: list[str] = []
        for index, (segment_text, requested_voice) in enumerate(segments):
            chunk = root / f"segment_{index:03d}.mp3"
            padded = root / f"segment_{index:03d}_padded.mp3"
            voice_candidates = [requested_voice]
            if requested_voice != preferred_voice:
                voice_candidates.append(preferred_voice)
            last_error = None
            segment_cues: list[dict] = []
            used_voice = requested_voice
            for voice in voice_candidates:
                try:
                    communicate = edge_tts.Communicate(segment_text, voice, rate=rate, pitch=pitch)
                    segment_cues = []
                    with open(chunk, "wb") as audio_file:
                        async for event in communicate.stream():
                            if event.get("type") == "audio":
                                audio_file.write(event["data"])
                            elif event.get("type") == "WordBoundary":
                                offset = float(event.get("offset", 0)) / 10_000_000
                                duration = float(event.get("duration", 0)) / 10_000_000
                                segment_cues.append({"start": offset, "end": offset + duration,
                                                     "text": event.get("text", "")})
                    if chunk.is_file() and chunk.stat().st_size >= 2000 and segment_cues:
                        used_voice = voice
                        break
                    raise RuntimeError("TTS segment returned no usable audio or word timings")
                except Exception as exc:
                    last_error = exc
                    if chunk.exists():
                        chunk.unlink()
            else:
                raise RuntimeError(f"TTS failed for a narration/dialogue segment: {last_error}")

            # Add a short breath after each segment. The added duration is included
            # in the next segment's word-boundary offsets.
            if index < len(segments) - 1:
                _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                      "-i", str(chunk), "-af", "apad=pad_dur=0.16",
                      "-c:a", "libmp3lame", "-q:a", "3", str(padded)], timeout=60)
                selected = padded
            else:
                selected = chunk
            duration = _duration(selected)
            for cue in segment_cues:
                word_cues.append({"start": cursor + cue["start"],
                                  "end": cursor + cue["end"], "text": cue["text"]})
            cursor += duration
            voice_sequence.append(used_voice)
            concat_lines.append("file '" + str(selected.resolve()).replace("'", "'\\''") + "'")

        concat_file = root / "segments.txt"
        concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
              "-f", "concat", "-safe", "0", "-i", str(concat_file),
              "-c:a", "libmp3lame", "-q:a", "3", str(target)], timeout=180)

    if not target.is_file() or target.stat().st_size < 5000:
        raise RuntimeError("Performed TTS did not produce a usable narration track")
    if not word_cues:
        raise RuntimeError("Performed TTS produced no word-boundary events")
    unique_voices = list(dict.fromkeys(voice_sequence))
    print(f"[VOICE] performed narration generated across {len(segments)} segments; voices: {', '.join(unique_voices)}")
    return ", ".join(unique_voices), word_cues

def _duration(path: Path) -> float:
    raw = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path)], 30).stdout.strip()
    value = float(raw)
    if value <= 0: raise RuntimeError("Media duration is invalid")
    return value

def _assemble(images: list[Path], beats: list[str], narration_audio: Path,
              output: Path, word_cues: list[dict]):
    """Render video with clean audio; export word-level captions as a separate SRT track."""
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
    caption_report = write_word_level_captions(word_cues, captions_path, audio_duration=duration)
    _run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(motion_video), "-i", str(narration_audio), "-i", str(sfx_path), "-i", str(music_path),
        "-filter_complex",
        "[1:a]aresample=48000,volume=1.0[narration];"
        "[2:a]aresample=48000,volume=0.34[sfx];"
        "[3:a]aresample=48000,volume=0.22[music];"
        "[narration][sfx][music]amix=inputs=3:duration=first:dropout_transition=0,"
        "alimiter=limit=0.92[audio]",
        "-map", "0:v:0", "-map", "[audio]", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-t", f"{duration:.3f}", "-movflags", "+faststart", str(output),
    ], timeout=360)
    if not output.is_file() or output.stat().st_size < 100_000:
        raise RuntimeError("FFmpeg did not produce a usable MP4 with narration, music and sound effects")
    return _duration(output), motion_report, sfx_report, music_report, caption_report, captions_path


def _generate_valid_story(client, model: str, topic: str, feedback: QualityFeedback,
                         recent_story_summaries: list[str] | None = None) -> dict:
    """Retry out-of-range drafts with progressively tighter word-count instructions."""
    last_error = None
    hook_style = HOOK_STYLES[(int(time.strftime("%j")) + len(topic or "")) % len(HOOK_STYLES)]
    # Story length is a minimum only. Never reject a coherent draft for being long.
    target_words = (180, 0)
    for attempt in range(3):
        brief = CreativeBrief(topic=topic, hook_style=hook_style,
                              duration_seconds=(50, 90), target_words=target_words)
        try:
            story = generate_story(client, model, brief, feedback, recent_story_summaries or [])
            story["hook_style"] = hook_style
            return story
        except ValueError as exc:
            if "words; minimum is" not in str(exc):
                raise
            last_error = exc
            print(f"[CREATIVE AI] Draft below {target_words[0]} words; requesting a fresh draft ({attempt + 1}/3)")
    raise ValueError(f"Creative AI could not produce a story of at least 180 words after three attempts: {last_error}")


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
    creative_model = (os.getenv("CREATIVE_AI_MODEL", "").strip() or "openai/gpt-oss-20b")
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
    voice_used, word_cues = asyncio.run(_synthesize_voice(story["narration"], audio_path))
    report["stages"].append({"name": "voice_generation", "status": "completed", "voice": voice_used, "rate": os.getenv("RELIC_LOOP_TTS_RATE", "+2%")})
    video_path = run_dir / "relic_loop_video.mp4"
    actual_duration, motion_report, sfx_report, music_report, caption_report, captions_path = _assemble(images, beats, audio_path, video_path, word_cues)
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
                           "captions": str(captions_path), "images": [str(x) for x in images]}
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
                patch = repair.get("patch", "")
                if (repair.get("repairable") and repair.get("category") == "code_bug"
                        and repair.get("confidence", 0) >= 0.8 and patch.strip()):
                    try:
                        repair["validation"] = maintenance_ai_validate(ROOT, patch)
                    except Exception as validation_error:
                        repair["validation"] = {"valid": False,
                            "reason": f"{type(validation_error).__name__}: {str(validation_error)[:500]}"}
                else:
                    repair["validation"] = {"valid": False, "reason": "No high-confidence code repair proposed"}
                repair["patch"] = patch
                failure["maintenance_ai"] = repair
                (OUT_ROOT / "maintenance_ai_report.json").write_text(
                    json.dumps(repair, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as diag:
            failure["maintenance_ai"] = {"diagnostic_error": str(diag)[:500], "deployment": "not_performed"}
        print(json.dumps(failure, indent=2))
        raise
