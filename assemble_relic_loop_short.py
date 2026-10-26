"""Assemble Relic Loop Shorts: narration is the master timeline; visuals follow it."""
from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

import edge_tts
from PIL import Image

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "state" / "current_run"
MANIFEST = RUN / "cloudflare_image_manifest.json"
STORY = RUN / "relic_loop_short.json"
FRAMES = RUN / "frame_clips"
OUTPUT = RUN / "relic_loop_short.mp4"
AUDIO = RUN / "voiceover.wav"
TTS_MP3 = RUN / "voiceover_neural.mp3"
FRAMES.mkdir(parents=True, exist_ok=True)

FPS = 30
W = 576
H = 1024
VOICE = __import__("os").environ.get("RELIC_LOOP_KOKORO_VOICE", "am_liam")
VOICE_PITCH = "+0Hz"
VOICE_VOLUME = "+0%"
TTS_RETRIES = 3
TTS_TIMEOUT_SECONDS = 120
MIN_SHORT_SECONDS = 60.0
MAX_SHORT_SECONDS = 90.0
TARGET_SHORT_SECONDS = 65.0


def run(cmd: list[str]) -> None:
    print("[FFMPEG]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def media_duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], text=True, capture_output=True, check=True)
    return float(result.stdout.strip())


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


async def _synthesize(text: str, rate: str) -> None:
    communicate = edge_tts.Communicate(text, VOICE, rate=rate, volume=VOICE_VOLUME, pitch=VOICE_PITCH)
    await asyncio.wait_for(communicate.save(str(TTS_MP3)), timeout=TTS_TIMEOUT_SECONDS)


def _rate_for_target(current_duration: float, target: float) -> str:
    """Estimate an Edge-TTS rate adjustment needed to move toward target duration."""
    adjustment = (current_duration / target - 1.0) * 100.0
    adjustment = max(-25.0, min(25.0, adjustment))
    return f"{int(round(adjustment)):+d}%"


def make_voiceover(text: str) -> float:
    text = _norm(text)
    if not text:
        raise SystemExit("Narration is empty; refusing to create a silent Short")
    if len(text.split()) < 50:
        raise SystemExit("Narration is suspiciously short; refusing to create a broken Short")

    print(f"[VOICE] Generating natural neural narration with {VOICE}")
    last: Exception | None = None
    rates: list[str] = ["+0%"]
    measured_duration: float | None = None

    for correction_attempt in range(3):
        rate = rates[-1]
        for attempt in range(1, TTS_RETRIES + 1):
            TTS_MP3.unlink(missing_ok=True)
            try:
                print(f"[VOICE] TTS attempt {attempt}/{TTS_RETRIES}; rate={rate}")
                asyncio.run(_synthesize(text, rate))
                if not TTS_MP3.exists() or TTS_MP3.stat().st_size < 1000:
                    raise RuntimeError("Neural TTS produced no usable audio")
                break
            except Exception as exc:
                last = exc
                print(f"[VOICE] TTS attempt failed: {type(exc).__name__}: {exc}")
                if attempt < TTS_RETRIES:
                    wait = 5 * attempt
                    print(f"[VOICE] Retrying TTS in {wait}s")
                    time.sleep(wait)
        else:
            raise SystemExit(f"Neural TTS failed after {TTS_RETRIES} attempts: {last}")

        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(TTS_MP3), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(AUDIO)])
        if not AUDIO.exists() or AUDIO.stat().st_size < 10000:
            raise SystemExit("Neural TTS conversion produced no usable WAV")

        measured_duration = media_duration(AUDIO)
        print(f"[VOICE] Candidate duration: {measured_duration:.2f}s at rate={rate}")
        if MIN_SHORT_SECONDS <= measured_duration <= MAX_SHORT_SECONDS:
            print(f"[VOICE] Ready: {measured_duration:.2f}s; inside required 60-90s target")
            return measured_duration

        if correction_attempt < 2:
            target = TARGET_SHORT_SECONDS if measured_duration < MIN_SHORT_SECONDS else 85.0
            next_rate = _rate_for_target(measured_duration, target)
            if next_rate == rate:
                next_rate = "-10%" if measured_duration < MIN_SHORT_SECONDS else "+10%"
            rates.append(next_rate)
            print(f"[VOICE] Duration outside 60-90s; retiming narration toward {target:.0f}s with rate={next_rate}")

    # Last-resort duration guard: if Kokoro's speed response is nonlinear or
    # a correction still misses the window, time-stretch the already-generated
    # audio locally. This uses no additional Cloudflare calls and prevents a
    # usable story from being discarded solely because narration is too short/long.
    if measured_duration is None or measured_duration <= 0:
        raise SystemExit("Narration duration could not be measured")
    target = min(MAX_SHORT_SECONDS, max(MIN_SHORT_SECONDS, TARGET_SHORT_SECONDS))
    tempo = measured_duration / target
    # FFmpeg atempo supports 0.5-2.0; this range is safely within those limits.
    tempo = max(0.75, min(1.25, tempo))
    print(f"[VOICE] Applying local atempo fallback: factor={tempo:.3f}; target={target:.0f}s")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(AUDIO),
         "-filter:a", f"atempo={tempo:.6f}", "-ac", "1", "-ar", "48000",
         "-c:a", "pcm_s16le", str(AUDIO.with_suffix(".retimed.wav"))])
    retimed = AUDIO.with_suffix(".retimed.wav")
    if not retimed.exists() or retimed.stat().st_size < 10000:
        raise SystemExit("Local narration retiming fallback produced no usable audio")
    retimed.replace(AUDIO)
    final_duration = media_duration(AUDIO)
    if not MIN_SHORT_SECONDS <= final_duration <= MAX_SHORT_SECONDS:
        raise SystemExit(f"Local narration retiming fallback still outside 60-90s: {final_duration:.2f}s")
    print(f"[VOICE] Local retiming fallback passed: {final_duration:.2f}s")
    return final_duration


def validate_image(image: Path, index: int) -> None:
    try:
        with Image.open(image) as im:
            im.verify()
        with Image.open(image) as im:
            if im.width < W or im.height < H:
                raise ValueError(f"image is only {im.width}x{im.height}; need at least {W}x{H}")
    except Exception as exc:
        raise SystemExit(f"Invalid Cloudflare image for beat {index}: {exc}") from exc


def _ease(frame: str, frames: int) -> str:
    """Smoothstep/ease-in-out progress with a literal frame denominator."""
    denom = max(frames - 1, 1)
    return f"(0.5-0.5*cos({frame}/{denom}*{math.pi:.12f}))"


def _camera_filter(profile: str, frames: int) -> str:
    """Return a story-oriented 2D camera move for one still image."""
    p = _ease("on", frames)
    margin = "(iw-iw/zoom)"
    vertical_margin = "(ih-ih/zoom)"

    if profile == "push_in":
        z = f"1.00+0.10*{p}"
        x = f"{margin}/2"
        y = f"{vertical_margin}/2"
    elif profile == "pull_out":
        z = f"1.11-0.09*{p}"
        x = f"{margin}/2"
        y = f"{vertical_margin}/2"
    elif profile == "pan_right":
        z = "1.07"
        x = f"{margin}*{p}"
        y = f"{vertical_margin}/2-0.025*{vertical_margin}*sin(on/{max(frames - 1, 1)}*3.14159265359)"
    elif profile == "pan_left":
        z = "1.07"
        x = f"{margin}*(1-{p})"
        y = f"{vertical_margin}/2+0.025*{vertical_margin}*sin(on/{max(frames - 1, 1)}*3.14159265359)"
    elif profile == "diagonal_right":
        z = f"1.03+0.08*{p}"
        x = f"{margin}*{p}"
        y = f"{vertical_margin}*(0.64-0.28*{p})"
    elif profile == "diagonal_left":
        z = f"1.03+0.08*{p}"
        x = f"{margin}*(1-{p})"
        y = f"{vertical_margin}*(0.36+0.28*{p})"
    elif profile == "punch_in":
        fast = f"min(on/({max(frames - 1, 1)}*0.22),1)"
        z = f"1.00+0.12*({fast})"
        x = f"{margin}/2"
        y = f"{vertical_margin}/2"
    elif profile == "reveal":
        z = f"1.12-0.10*{p}"
        x = f"{margin}*(0.78-0.28*{p})"
        y = f"{vertical_margin}*(0.35+0.30*{p})"
    else:
        z = f"1.04+0.025*{p}"
        x = f"{margin}/2+0.10*{margin}*sin(on/{max(frames - 1, 1)}*6.28318530718)"
        y = f"{vertical_margin}/2+0.07*{vertical_margin}*sin(on/{max(frames - 1, 1)}*4.71238898038)"

    return (
        f"scale=ceil(iw*1.25/2)*2:ceil(ih*1.25/2)*2,"
        f"zoompan=z='{z}':x='{x}':y='{y}':"
        f"d={frames}:s={W}x{H}:fps={FPS},format=yuv420p"
    )


def _camera_profile(index: int, duration: float, action: str, mood: str) -> str:
    text = f"{action} {mood}".lower()
    if any(word in text for word in ("reveal", "discover", "realize", "twist", "surprise")):
        return "reveal"
    if any(word in text for word in ("run", "chase", "rush", "move", "walk", "drive", "escape")):
        return "pan_right" if index % 2 else "pan_left"
    if any(word in text for word in ("shock", "panic", "reaction", "confused", "angry", "excited")):
        return "punch_in"
    if any(word in text for word in ("look", "watch", "notice", "wait", "think")):
        return "push_in"
    profiles = ("push_in", "pan_right", "pull_out", "diagonal_left", "pan_left", "punch_in", "diagonal_right", "handheld")
    return profiles[(index - 1) % len(profiles)]


def make_clip(image: Path, duration: float, index: int, action: str = "", mood: str = "") -> Path:
    out = FRAMES / f"clip_{index:03d}.mp4"
    frames = max(6, round(duration * FPS))
    exact_duration = frames / FPS
    profile = _camera_profile(index, duration, action, mood)
    vf = _camera_filter(profile, frames)
    print(f"[CAMERA] beat {index}: profile={profile}; duration={exact_duration:.2f}s")
    try:
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i", str(image), "-vf", vf, "-frames:v", str(frames), "-an", "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-video_track_timescale", "30000", str(out)])
    except subprocess.CalledProcessError as exc:
        out.unlink(missing_ok=True)
        raise SystemExit(f"Camera-motion render failed on beat {index} (profile={profile}); refusing to silently fall back to a static shot") from exc
    measured = media_duration(out)
    if abs(measured - exact_duration) > 0.08:
        raise SystemExit(f"Beat {index} timing error: requested={exact_duration:.3f}s, clip={measured:.3f}s")
    return out


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", _norm(text)))


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit("Missing cloudflare_image_manifest.json")
    if not STORY.exists():
        raise SystemExit("Missing relic_loop_short.json")

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    story = json.loads(STORY.read_text(encoding="utf-8"))
    beats = data.get("beats") or []
    narration = _norm(story.get("narration", ""))
    if not beats:
        raise SystemExit("Story contains no rendered visual beats")
    if _word_count(narration) < 50:
        raise SystemExit("Master narration is too short")

    audio_duration = make_voiceover(narration)
    total_words = _word_count(narration)

    beat_word_counts = [_word_count(str(b.get("narration", ""))) for b in beats]
    if sum(beat_word_counts) != total_words:
        raise SystemExit(f"Beat narration does not cover the complete story: beat_words={sum(beat_word_counts)}, master_words={total_words}. Refusing to make a video where visuals and narration drift apart.")
    if any(count <= 0 for count in beat_word_counts):
        raise SystemExit("Every visual beat must contain narration text")

    raw_durations = [audio_duration * count / total_words for count in beat_word_counts]
    durations = [max(0.20, d) for d in raw_durations]
    scale = audio_duration / sum(durations)
    durations = [d * scale for d in durations]

    shutil.rmtree(FRAMES, ignore_errors=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    actual_clip_durations: list[float] = []
    for i, (beat, duration) in enumerate(zip(beats, durations), 1):
        image = Path(beat["image"])
        if not image.exists():
            raise SystemExit(f"Missing image for beat {i}: {image}")
        validate_image(image, i)
        clip = make_clip(image, duration, i, str(beat.get("action", "")), str(beat.get("mood", "")))
        clips.append(clip)
        actual_clip_durations.append(media_duration(clip))
        print(f"[TIMING] beat {i}: {beat_word_counts[i-1]} words -> target {duration:.2f}s -> clip {actual_clip_durations[-1]:.2f}s")

    concat = RUN / "clips.txt"
    concat.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in clips) + "\n", encoding="utf-8")
    silent = RUN / "silent_short.mp4"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-an", "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-video_track_timescale", "30000", str(silent)])
    if not silent.exists() or silent.stat().st_size == 0:
        raise SystemExit("Silent video assembly produced no MP4")

    video_duration = media_duration(silent)
    print(f"[TIMING] visual timeline={video_duration:.3f}s; narration master={audio_duration:.3f}s")
    if abs(video_duration - audio_duration) > 0.35:
        raise SystemExit(f"Story timing mismatch: narration={audio_duration:.2f}s, visual timeline={video_duration:.2f}s")

    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(AUDIO), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", f"{audio_duration:.3f}", "-movflags", "+faststart", str(OUTPUT)])
    if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
        raise SystemExit("Final video assembly produced no MP4")

    final_duration = media_duration(OUTPUT)
    if abs(final_duration - audio_duration) > 0.35:
        raise SystemExit(f"Final duration mismatch: narration={audio_duration:.2f}s, final={final_duration:.2f}s")
    if not MIN_SHORT_SECONDS <= final_duration <= MAX_SHORT_SECONDS:
        raise SystemExit(f"Final MP4 duration {final_duration:.2f}s is outside the required 60-90s range after narration retiming")
    if final_duration < 5:
        raise SystemExit("Final MP4 is unusably short")

    timing = {
        "timing_policy": "narration_is_master_timeline",
        "duration_policy": "neural_narration_is_automatically_retimed_to_60_90_seconds_before_visual_assembly",
        "image_count_policy": "10_to_14_phrase_aligned_beats; durations follow narration segment word share; no fixed seconds per image",
        "camera_policy": "story_aware_2d_camera_profiles",
        "master_narration_seconds": audio_duration,
        "master_narration_words": total_words,
        "beats": [{"number": i, "narration_words": beat_word_counts[i-1], "target_duration_seconds": round(durations[i-1], 3), "clip_duration_seconds": round(actual_clip_durations[i-1], 3), "camera_profile": _camera_profile(i, durations[i-1], str(beats[i-1].get("action", "")), str(beats[i-1].get("mood", ""))), "image": beats[i-1]["image"]} for i in range(1, len(beats) + 1)],
        "visual_timeline_seconds": video_duration,
        "final_duration_seconds": final_duration,
    }
    (RUN / "timing_manifest.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"[SHORT] READY: {OUTPUT}")
    print(f"[SHORT] narration-master duration: {audio_duration:.2f}s")
    print(f"[SHORT] final duration: {final_duration:.2f}s")


if __name__ == "__main__":
    main()
