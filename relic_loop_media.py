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
import textwrap
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
    frame_counts = [max(2, round(max(0.25, float(seconds)) * fps)) for seconds in durations]
    target_frames = max(2, round(sum(max(0.25, float(seconds)) for seconds in durations) * fps))
    frame_counts[-1] = max(2, frame_counts[-1] + target_frames - sum(frame_counts))

    for index, (image, seconds) in enumerate(zip(images, durations)):
        seconds = max(0.25, float(seconds))
        frames = frame_counts[index]
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
    pcm = array.array("h", mix)
    if sys.byteorder != "little":
        pcm.byteswap()
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
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
        music_path = root / "music.wav"
        music = build_background_music_track([0.6, 0.6], music_path, mood="mysterious")
        assert music_path.is_file() and music_path.stat().st_size > 1000
        captions_path = root / "captions.srt"
        captions = write_scene_captions(
            ["A mysterious letter appeared.", "The secret was finally revealed."],
            [0.6, 0.6], captions_path)
        assert captions_path.is_file() and captions["caption_cues"] == 2
        assert "-->" in captions_path.read_text(encoding="utf-8")
    print("Relic Loop local motion/audio self-test passed")

if __name__ == "__main__":
    self_test()


def build_background_music_track(
    durations: Sequence[float],
    output_path: Path,
    mood: str = "playful",
    sample_rate: int = 48000,
) -> dict:
    """Create a gentle, original instrumental bed; no stock track or external API."""
    if not durations or any(float(x) <= 0 for x in durations):
        raise ValueError("Background music requires positive scene durations")
    total = sum(float(x) for x in durations)
    count = max(1, int(math.ceil(total * sample_rate)))
    # Soft triad progression; playful stories use a brighter major palette.
    progressions = {
        "playful": [(261.63, 329.63, 392.00), (293.66, 369.99, 440.00),
                    (329.63, 392.00, 493.88), (261.63, 329.63, 392.00)],
        "mysterious": [(220.00, 261.63, 329.63), (196.00, 233.08, 293.66),
                       (220.00, 261.63, 311.13), (196.00, 233.08, 293.66)],
        "warm": [(261.63, 329.63, 392.00), (349.23, 440.00, 523.25),
                 (293.66, 369.99, 440.00), (261.63, 329.63, 392.00)],
        "emotional": [(220.00, 261.63, 329.63), (174.61, 220.00, 261.63),
                      (196.00, 246.94, 293.66), (220.00, 261.63, 329.63)],
    }
    chords = progressions.get(mood, progressions["playful"])
    samples = array.array("h")
    fade_in = max(1, int(sample_rate * 0.8))
    fade_out = max(1, int(sample_rate * 1.8))
    for i in range(count):
        t = i / sample_rate
        chord_index = min(3, int(t / max(0.01, total / 4)))
        chord = chords[chord_index]
        # Very soft layered sine pads, with slow modulation to avoid a static beep.
        pad = sum(math.sin(2 * math.pi * f * t) for f in chord) / 3
        shimmer = math.sin(2 * math.pi * (chord[1] * 2) * t) * 0.08
        envelope = min(1.0, i / fade_in, (count - i) / fade_out)
        value = (pad * 0.72 + shimmer) * envelope * 0.035
        samples.append(max(-32767, min(32767, int(value * 32767))))
    pcm = array.array("h", samples)
    if sys.byteorder != "little":
        pcm.byteswap()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return {"mood": mood, "duration_seconds": round(total, 2),
            "source": "locally_synthesized_original_music", "target_level": "very_low_under_narration"}


def write_scene_captions(beats: Sequence[str], durations: Sequence[float], output_path: Path) -> dict:
    """Write readable, sentence-aware SRT captions synchronized to scene timing."""
    if not beats or len(beats) != len(durations):
        raise ValueError("Captions require matching non-empty beat and duration lists")
    def stamp(seconds: float) -> str:
        ms = max(0, round(seconds * 1000))
        hours, ms = divmod(ms, 3_600_000)
        minutes, ms = divmod(ms, 60_000)
        secs, ms = divmod(ms, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    entries, cursor = [], 0.0
    for idx, (beat, seconds) in enumerate(zip(beats, durations), 1):
        clean = re.sub(r"\s+", " ", str(beat)).strip()
        if not clean:
            cursor += float(seconds); continue
        # Keep each caption concise and split long narration beats into readable cards.
        words = clean.split()
        chunks = [" ".join(words[i:i + 9]) for i in range(0, len(words), 9)]
        span = float(seconds) / max(1, len(chunks))
        for chunk in chunks:
            start, end = cursor, cursor + span
            entries.append(f"{len(entries)+1}\n{stamp(start)} --> {stamp(end)}\n{chunk}\n")
            cursor = end
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(entries), encoding="utf-8")
    return {"caption_cues": len(entries), "format": "srt",
            "max_words_per_caption": 9, "synchronized_to_scene_timing": True}
