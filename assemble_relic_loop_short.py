"""Assemble generated Relic Loop image beats into a vertical MP4 with robust camera motion."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "state" / "current_run"
MANIFEST = RUN / "cloudflare_image_manifest.json"
FRAMES = RUN / "frame_clips"
OUTPUT = RUN / "relic_loop_short.mp4"
AUDIO = RUN / "voiceover.wav"
FRAMES.mkdir(parents=True, exist_ok=True)

FPS = 30
W = 576
H = 1024


def run(cmd: list[str]) -> None:
    print("[FFMPEG]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def make_voiceover(text: str) -> None:
    if shutil.which("espeak-ng") is None:
        raise SystemExit("espeak-ng is required")
    run(["espeak-ng", "-v", "en-us+m3", "-s", "178", "-p", "48", "-a", "145", "-w", str(AUDIO), text])
    if not AUDIO.exists() or AUDIO.stat().st_size == 0:
        raise SystemExit("Voiceover generation produced no audio file")


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

    # IMPORTANT: do not crop using iw/ih after scaling. The previous filter produced
    # invalid crop dimensions on FFmpeg and killed the entire run. Zoompan itself can
    # perform the center crop safely once the source is guaranteed larger than output.
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
        # A single camera-motion expression must never destroy an otherwise valid run.
        # Retry the same frame with a deliberately boring but reliable static filter.
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
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    beats = data.get("beats") or []
    if len(beats) < 24:
        raise SystemExit(f"Refusing to assemble only {len(beats)} beats; minimum is 24")

    shutil.rmtree(FRAMES, ignore_errors=True)
    FRAMES.mkdir(parents=True, exist_ok=True)

    clips: list[Path] = []
    for i, beat in enumerate(beats, 1):
        image = Path(beat["image"])
        if not image.exists():
            raise SystemExit(f"Missing image for beat {i}: {image}")
        validate_image(image, i)
        duration = float(beat["duration_seconds"])
        if not 0.8 <= duration <= 1.8:
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

    make_voiceover(data.get("narration", ""))
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(silent), "-i", str(AUDIO),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(OUTPUT),
    ])
    if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
        raise SystemExit("Final video assembly produced no MP4")
    print(f"[SHORT] READY: {OUTPUT}")


if __name__ == "__main__":
    main()
