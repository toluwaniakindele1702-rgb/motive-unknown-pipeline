"""Assemble generated Relic Loop story beats into a vertical MP4 with natural narration and robust camera motion."""
from __future__ import annotations

import asyncio
import json
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


def run(cmd: list[str]) -> None:
    print("[FFMPEG]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def media_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True, capture_output=True, check=True,
    )
    return float(result.stdout.strip())


async def _synthesize(text: str) -> None:
    communicate = edge_tts.Communicate(
        text,
        VOICE,
        rate=VOICE_RATE,
        volume=VOICE_VOLUME,
        pitch=VOICE_PITCH,
    )
    await communicate.save(str(TTS_MP3))


def make_voiceover(text: str) -> None:
    text = " ".join(str(text or "").split())
    if not text:
        raise SystemExit("Narration is empty; refusing to create a silent Short")
    if len(text.split()) < 50:
        raise SystemExit("Narration is suspiciously short; refusing to create a silent/empty Short")

    print(f"[VOICE] Generating natural neural narration with {VOICE}")
    TTS_MP3.unlink(missing_ok=True)
    asyncio.run(_synthesize(text))
    if not TTS_MP3.exists() or TTS_MP3.stat().st_size < 1000:
        raise SystemExit("Neural TTS produced no usable audio")

    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(TTS_MP3), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(AUDIO),
    ])
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
    out = FRAMES / f"clip_{index:03d}.mp4"
    frames = max(1, round(duration * FPS))

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
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-loop", "1", "-i", str(image), "-vf", vf,
            "-frames:v", str(frames), "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-pix_fmt", "yuv420p", str(out),
        ])
    except subprocess.CalledProcessError:
        print(f"[ASSEMBLY] camera-motion filter failed on beat {index}; using safe static fallback")
        out.unlink(missing_ok=True)
        fallback = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-loop", "1", "-i", str(image), "-vf", fallback,
            "-frames:v", str(frames), "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-pix_fmt", "yuv420p", str(out),
        ])
    return out


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit("Missing cloudflare_image_manifest.json")
    if not STORY.exists():
        raise SystemExit("Missing relic_loop_short.json")

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    story = json.loads(STORY.read_text(encoding="utf-8"))
    beats = data.get("beats") or []
    if not 12 <= len(beats) <= 16:
        raise SystemExit(f"Refusing to assemble {len(beats)} beats; expected 12-16")

    shutil.rmtree(FRAMES, ignore_errors=True)
    FRAMES.mkdir(parents=True, exist_ok=True)

    clips: list[Path] = []
    for i, beat in enumerate(beats, 1):
        image = Path(beat["image"])
        if not image.exists():
            raise SystemExit(f"Missing image for beat {i}: {image}")
        validate_image(image, i)
        duration = float(beat["duration_seconds"])
        if not 1.4 <= duration <= 3.2:
            raise SystemExit(f"Invalid beat {i} duration: {duration}")
        clips.append(make_clip(image, duration, i))

    concat = RUN / "clips.txt"
    concat.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in clips) + "\n", encoding="utf-8")
    silent = RUN / "silent_short.mp4"
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat),
        "-fflags", "+genpts", "-c", "copy", str(silent),
    ])
    if not silent.exists() or silent.stat().st_size == 0:
        raise SystemExit("Silent video assembly produced no MP4")

    make_voiceover(story.get("narration", ""))
    video_duration = media_duration(silent)
    audio_duration = media_duration(AUDIO)
    if audio_duration < video_duration * 0.82 or audio_duration > video_duration * 1.18:
        raise SystemExit(
            f"Narration/video timing mismatch: narration={audio_duration:.2f}s, video={video_duration:.2f}s; "
            "refusing to publish a badly synced Short"
        )

    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(silent), "-i", str(AUDIO),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(OUTPUT),
    ])
    if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
        raise SystemExit("Final video assembly produced no MP4")

    final_audio = media_duration(OUTPUT)
    if final_audio < 5:
        raise SystemExit("Final MP4 audio track is missing or unusably short")
    print(f"[SHORT] READY: {OUTPUT}")
    print(f"[SHORT] final duration: {final_audio:.2f}s")


if __name__ == "__main__":
    main()
