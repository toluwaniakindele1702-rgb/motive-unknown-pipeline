"""Assemble generated Relic Loop image beats into a vertical MP4 with subtle camera motion."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

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
    # A brisk, clear voice; this is only the fallback Shorts-flow voice.
    run(["espeak-ng", "-v", "en-us+m3", "-s", "178", "-p", "48", "-a", "145", "-w", str(AUDIO), text])


def make_clip(image: Path, duration: float, index: int) -> Path:
    out = FRAMES / f"clip_{index:03d}.mp4"
    frames = max(1, round(duration * FPS))
    # Alternate a gentle push-in and pull-out. This creates actual camera motion even
    # though the source is a still image; cuts remain immediate between beats.
    if index % 2:
        zoom = "min(zoom+0.0018,1.07)"
    else:
        zoom = "if(lte(zoom,1.0),1.07,max(zoom-0.0018,1.0))"
    vf = (
        f"scale=iw*1.12:ih*1.12:force_original_aspect_ratio=increase," 
        f"crop=ceil(iw/2)*2:ceil(ih/2)*2," 
        f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s={W}x{H}:fps={FPS},format=yuv420p"
    )
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-i", str(image), "-vf", vf,
        "-frames:v", str(frames), "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "22", "-pix_fmt", "yuv420p", str(out),
    ])
    return out


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit("Missing cloudflare_image_manifest.json")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    beats = data.get("beats") or []
    if len(beats) < 20:
        raise SystemExit("Refusing to assemble fewer than 20 beats")

    shutil.rmtree(FRAMES, ignore_errors=True)
    FRAMES.mkdir(parents=True, exist_ok=True)

    clips: list[Path] = []
    for i, beat in enumerate(beats, 1):
        image = Path(beat["image"])
        if not image.exists():
            raise SystemExit(f"Missing image for beat {i}: {image}")
        clips.append(make_clip(image, float(beat["duration_seconds"]), i))

    concat = RUN / "clips.txt"
    concat.write_text("\n".join(f"file '{p.as_posix()}'" for p in clips) + "\n", encoding="utf-8")
    silent = RUN / "silent_short.mp4"
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat),
        "-c", "copy", str(silent),
    ])

    make_voiceover(data.get("narration", ""))
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(silent), "-i", str(AUDIO),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(OUTPUT),
    ])
    print(f"[SHORT] READY: {OUTPUT}")


if __name__ == "__main__":
    main()
