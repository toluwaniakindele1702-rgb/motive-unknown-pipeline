#!/usr/bin/env python3
"""Local, deterministic camera motion and lightweight sound design for Relic Loop.

No camera website, browser automation, external sound library, or paid media API is
used. FFmpeg renders image motion; short sound cues are synthesized locally.
"""
from __future__ import annotations

import array
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Sequence


def _run(command: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(
            f"Media command failed ({result.returncode}): {' '.join(command[:6])}; "
            f"{(result.stderr or '')[-1600:]}"
        )
    return result


def _motion_expression(index: int, frames: int) -> tuple[str, str, str, str]:
    """Return a gentle, alternating motion preset and its FFmpeg expressions."""
    t = f"on/{max(1, frames - 1)}"
    presets = [
        ("gentle_push_in", f"1+0.11*{t}", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"),
        ("gentle_pull_out", f"1.14-0.10*{t}", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"),
        ("pan_left_to_right", "1.14", f"(iw-iw/zoom)*(0.16+0.68*{t})", "(ih-ih/zoom)/2"),
        ("pan_right_to_left", "1.14", f"(iw-iw/zoom)*(0.84-0.68*{t})", "(ih-ih/zoom)/2"),
        ("diagonal_reveal", f"1.08+0.08*{t}",
         f"(iw-iw/zoom)*(0.30+0.35*{t})",
         f"(ih-ih/zoom)*(0.66-0.32*{t})"),
    ]
    return presets[index % len(presets)]


def render_motion_video(
    images: Sequence[Path],
    durations: Sequence[float],
    output_path: Path,
    width: int = 576,
    height: int = 1024,
    fps: int = 30,
) -> list[dict]:
    """Render each still as a real moving clip, then concatenate the clips."""
    if not images or len(images) != len(durations):
        raise ValueError("Camera motion requires matching non-empty image and duration lists")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for local camera motion rendering")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip_dir = output_path.parent / "motion_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    manifest = clip_dir / "clips.concat.txt"
    concat_lines: list[str] = []
    motion_report: list[dict] = []

    for index, (image, seconds) in enumerate(zip(images, durations)):
        seconds = max(0.25, float(seconds))
        frames = max(2, round(seconds * fps))
        name, zoom, x_expr, y_expr = _motion_expression(index, frames)
        clip = clip_dir / f"scene_{index + 1:02d}.mp4"
        # Supersampling before zoompan reduces integer-pixel stepping. The
        # viewport is always cropped to the target aspect ratio before motion.
        vf = (
            f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,"
            f"crop={width * 2}:{height * 2},"
            f"zoompan=z='{zoom}':x='{x_expr}':y='{y_expr}':d=1:"
            f"s={width}x{height}:fps={fps},format=yuv420p"
        )
        _run([
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-loop", "1", "-framerate", str(fps), "-i", str(Path(image).resolve()),
            "-vf", vf, "-frames:v", str(frames), "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(fps), str(clip),
        ], timeout=max(90, int(seconds * 20)))
        if not clip.is_file() or clip.stat().st_size < 1000:
            raise RuntimeError(f"Camera motion render produced no usable clip for scene {index + 1}")
        escaped = str(clip.resolve()).replace("'", "'\\''")
        concat_lines.append(f"file '{escaped}'")
        motion_report.append({
            "scene": index + 1, "motion": name, "duration_seconds": round(frames / fps, 3),
            "frames": frames, "supersampling": "2x",
        })

    manifest.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
    _run([
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(manifest),
        "-c", "copy", "-movflags", "+faststart", str(output_path),
    ], timeout=180)
    if not output_path.is_file() or output_path.stat().st_size < 10_000:
        raise RuntimeError("Camera motion clips could not be assembled")
    return motion_report


def _synthesize_effect(kind: str, sample_rate: int = 48000) -> array.array:
    """Create a short, quiet, non-verbal sound cue as signed 16-bit PCM."""
    duration = {"whoosh": 0.42, "pop": 0.22, "chime": 0.62}.get(kind, 0.35)
    count = int(sample_rate * duration)
    samples = array.array("h")
    rng = random.Random(1907 + len(kind))
    previous_noise = 0.0
    for i in range(count):
        t = i / sample_rate
        p = i / max(1, count - 1)
        if kind == "whoosh":
            noise = rng.uniform(-1.0, 1.0)
            filtered = noise - previous_noise * 0.72
            previous_noise = noise
            envelope = math.sin(math.pi * p) ** 1.7
            value = filtered * envelope * 0.12
        elif kind == "pop":
            low = math.sin(2 * math.pi * (105 - 48 * p) * t) * math.exp(-8 * p)
            click = math.sin(2 * math.pi * 1150 * t) * math.exp(-65 * p)
            value = (low * 0.13 + click * 0.035)
        else:
            envelope = math.exp(-4.2 * p)
            tone = (
                math.sin(2 * math.pi * 880 * t)
                + 0.62 * math.sin(2 * math.pi * 1174.66 * t)
                + 0.32 * math.sin(2 * math.pi * 1568 * t)
            ) / 1.94
            value = tone * envelope * 0.095
        # Gentle edge fades prevent clicks at cue boundaries.
        fade = min(1.0, p * 30, (1.0 - p) * 35)
        value *= max(0.0, fade)
        samples.append(max(-32767, min(32767, int(value * 32767))))
    return samples


def build_sound_effect_track(
    beats: Sequence[str],
    durations: Sequence[float],
    output_path: Path,
    sample_rate: int = 48000,
) -> list[dict]:
    """Synthesize sparse story-aware cues into one full-length WAV track."""
    if len(beats) != len(durations) or not beats:
        raise ValueError("Sound design requires matching non-empty beat and duration lists")
    total_seconds = sum(max(0.0, float(x)) for x in durations)
    total_samples = max(1, int(math.ceil(total_seconds * sample_rate)))
    mix = array.array("h", [0]) * total_samples
    if sys.byteorder != "little":
        mix.byteswap()

    cues: list[tuple[float, str, int]] = []
    elapsed = 0.0
    for index, (beat, seconds) in enumerate(zip(beats, durations)):
        text = str(beat).lower()
        kind = None
        if index == 0:
            kind = "whoosh"
        if re.search(r"\b(reveal|revealed|secret|truth|realized|finally|discovered|found out)\b", text):
            kind = "chime"
        elif re.search(r"\b(suddenly|dropped|slam|crash|burst|knocked|grabbed|snatched|opened)\b", text):
            kind = "pop"
        elif re.search(r"\b(looked|turned|noticed|found|saw)\b", text) and index % 2 == 0:
            kind = "whoosh"
        cue_time = elapsed + min(0.16, max(0.04, float(seconds) * 0.04))
        if kind and (not cues or cue_time - cues[-1][0] >= 4.0):
            cues.append((cue_time, kind, index + 1))
        elapsed += float(seconds)

    # Always provide a soft closing accent when the script has no obvious reveal word.
    if total_seconds > 8 and (not cues or cues[-1][2] != len(beats)):
        cues.append((max(0.0, total_seconds - 0.48), "chime", len(beats)))
    cues = cues[:5]

    cue_report = []
    for cue_time, kind, scene in cues:
        effect = _synthesize_effect(kind, sample_rate)
        start = min(total_samples, int(cue_time * sample_rate))
        for offset, sample in enumerate(effect):
            target = start + offset
            if target >= total_samples:
                break
            mixed = int(mix[target]) + int(sample * 0.78)
            mix[target] = max(-32767, min(32767, mixed))
        cue_report.append({"scene": scene, "time_seconds": round(cue_time, 2), "effect": kind, "level": "subtle"})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(mix.tobytes())
    return cue_report


def self_test() -> None:
    """Render a tiny moving clip and WAV offline to catch broken media wiring."""
    for index in range(5):
        name, zoom, x, y = _motion_expression(index, 30)
        assert name and zoom and x and y
    cue = _synthesize_effect("chime")
    assert len(cue) > 1000 and max(cue) > 0
    with tempfile.TemporaryDirectory(prefix="relic-loop-media-test-") as folder:
        root = Path(folder)
        still = root / "test.png"
        _run([
            shutil.which("ffmpeg") or "ffmpeg", "-nostdin", "-hide_banner",
            "-loglevel", "error", "-y", "-f", "lavfi", "-i",
            "color=c=0x456789:s=288x512", "-frames:v", "1", str(still),
        ])
        rendered = root / "motion.mp4"
        report = render_motion_video([still], [0.6], rendered, width=144, height=256, fps=15)
        assert rendered.is_file() and report[0]["motion"] == "gentle_push_in"
        wav_path = root / "sfx.wav"
        cues = build_sound_effect_track(["A mysterious letter appeared.", "The secret was revealed."],
                                        [0.6, 0.6], wav_path)
        assert wav_path.is_file() and wav_path.stat().st_size > 1000 and cues
    print("Relic Loop local motion/audio self-test passed")

if __name__ == "__main__":
    self_test()
