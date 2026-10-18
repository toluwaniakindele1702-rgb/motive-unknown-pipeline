"""Assemble Relic Loop Shorts: narration is the master timeline; visuals follow it."""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
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
VOICE = "en-US-AndrewNeural"
VOICE_RATE = "+0%"
VOICE_PITCH = "+0Hz"
VOICE_VOLUME = "+0%"
TTS_RETRIES = 3
TTS_TIMEOUT_SECONDS = 120


def run(cmd: list[str]) -> None:
    print("[FFMPEG]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def media_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True, capture_output=True, check=True,
    )
    return float(result.stdout.strip())


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


async def _synthesize(text: str) -> None:
    communicate = edge_tts.Communicate(text, VOICE, rate=VOICE_RATE, volume=VOICE_VOLUME, pitch=VOICE_PITCH)
    await asyncio.wait_for(communicate.save(str(TTS_MP3)), timeout=TTS_TIMEOUT_SECONDS)


def make_voiceover(text: str) -> None:
    text = _norm(text)
    if not text:
        raise SystemExit("Narration is empty; refusing to create a silent Short")
    if len(text.split()) < 50:
        raise SystemExit("Narration is suspiciously short; refusing to create a broken Short")

    print(f"[VOICE] Generating natural neural narration with {VOICE}")
    last: Exception | None = None
    for attempt in range(1, TTS_RETRIES + 1):
        TTS_MP3.unlink(missing_ok=True)
        try:
            print(f"[VOICE] TTS attempt {attempt}/{TTS_RETRIES}")
            asyncio.run(_synthesize(text))
            if not TTS_MP3.exists() or TTS_MP3.stat().st_size < 1000:
                raise RuntimeError("Neural TTS produced no usable audio")
            break
        except Exception as exc:
            last = exc
            print(f"[VOICE] TTS attempt failed: {type(exc).__name__}: {exc}")
            if attempt < TTS_RETRIES:
                wait = 5 * attempt
                print(f"[VOICE] Retrying TTS in {wait}s")
                import time
                time.sleep(wait)
    else:
        raise SystemExit(f"Neural TTS failed after {TTS_RETRIES} attempts: {last}")

    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(TTS_MP3), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(AUDIO)])
    if not AUDIO.exists() or AUDIO.stat().st_size < 10000:
        raise SystemExit("Neural TTS conversion produced no usable WAV")
    duration = media_duration(AUDIO)
    if duration < 5:
        raise SystemExit(f"Neural narration is only {duration:.2f}s; refusing to publish a broken Short")
    print(f"[VOICE] Ready: {duration:.2f}s")


def validate_image(image: Path, index: int) -> None:
    try:
        with Image.open(image) as im:
            im.verify()
        with Image.open(image) as im:
            if im.width < W or im.height < H:
                raise ValueError(f"image is only {im.width}x{im.height}; need at least {W}x{H}")
    except Exception as exc:
        raise SystemExit(f"Invalid Cloudflare image for beat {index}: {exc}") from exc


def make_clip(image: Path, duration: float, index: int) -> Path:
    """Create a CFR clip whose measured duration is the requested beat duration."""
    out = FRAMES / f"clip_{index:03d}.mp4"
    frames = max(6, round(duration * FPS))
    exact_duration = frames / FPS
    if index % 2:
        zoom = "min(zoom+0.0018,1.07)"
    else:
        zoom = "if(lte(zoom,1.0),1.07,max(zoom-0.0018,1.0))"
    vf = (
        f"scale=ceil(iw*1.12/2)*2:ceil(ih*1.12/2)*2,"
        f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s={W}x{H}:fps={FPS},format=yuv420p"
    )
    try:
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i", str(image), "-vf", vf, "-frames:v", str(frames), "-an", "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-video_track_timescale", "30000", str(out)])
    except subprocess.CalledProcessError:
        print(f"[ASSEMBLY] camera-motion filter failed on beat {index}; using safe static fallback")
        out.unlink(missing_ok=True)
        fallback = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i", str(image), "-vf", fallback, "-frames:v", str(frames), "-an", "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-video_track_timescale", "30000", str(out)])
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
    if not 10 <= len(beats) <= 18:
        raise SystemExit(f"Refusing to assemble {len(beats)} beats; expected 10-18")
    if _word_count(narration) < 50:
        raise SystemExit("Master narration is too short")

    # Generate the actual narration FIRST. This duration is the master video duration.
    make_voiceover(narration)
    audio_duration = media_duration(AUDIO)
    total_words = _word_count(narration)

    # Scene narration is an ordered partition of the master narration. Allocate each image
    # a proportional share of the real TTS duration. There is no fixed image timer.
    beat_word_counts = [_word_count(str(b.get("narration", ""))) for b in beats]
    if sum(beat_word_counts) != total_words:
        raise SystemExit(
            f"Beat narration does not cover the complete story: beat_words={sum(beat_word_counts)}, master_words={total_words}. "
            "Refusing to make a video where visuals and narration drift apart."
        )
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
        clip = make_clip(image, duration, i)
        clips.append(clip)
        actual_clip_durations.append(media_duration(clip))
        print(f"[TIMING] beat {i}: {beat_word_counts[i-1]} words -> target {duration:.2f}s -> clip {actual_clip_durations[-1]:.2f}s")

    # Rebuild the timeline with FFmpeg re-encoding instead of concat stream-copy. Stream-copy
    # can inherit timestamp rounding/offsets and make the combined visual timeline shorter.
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
    if final_duration < 5:
        raise SystemExit("Final MP4 is unusably short")

    timing = {
        "timing_policy": "narration_is_master_timeline",
        "master_narration_seconds": audio_duration,
        "master_narration_words": total_words,
        "beats": [
            {"number": i, "narration_words": beat_word_counts[i-1], "target_duration_seconds": round(durations[i-1], 3), "clip_duration_seconds": round(actual_clip_durations[i-1], 3), "image": beats[i-1]["image"]}
            for i in range(1, len(beats) + 1)
        ],
        "visual_timeline_seconds": video_duration,
        "final_duration_seconds": final_duration,
    }
    (RUN / "timing_manifest.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"[SHORT] READY: {OUTPUT}")
    print(f"[SHORT] narration-master duration: {audio_duration:.2f}s")
    print(f"[SHORT] final duration: {final_duration:.2f}s")


if __name__ == "__main__":
    main()
