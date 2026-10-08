"""Relic Loop V2 isolated 2-minute prototype.

Test target: "Why do you forget why you walked into a room?"
- ~2 minutes
- 60% still-image storytelling / 40% Veo animation
- recurring RL character generated once, then reused as an image reference
- automated camera movement on stills
- Veo failures immediately fall back to the still image
- no subtitles or baked-in scene text
- does not upload to YouTube and does not touch the production queue
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from veo_v2 import animate_or_fallback, generate_image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "run_work" / "v2_test"
IMG = OUT / "images"
VID = OUT / "videos"
AUD = OUT / "audio"
for d in (IMG, VID, AUD):
    d.mkdir(parents=True, exist_ok=True)

RL_SPEC = json.loads((ROOT / "assets/characters/rl/rl_character.json").read_text(encoding="utf-8"))
RL_PROMPT = RL_SPEC["generation_prompt"]

SCENES = [
    ("image", "Have you ever walked into a room and instantly forgotten why you went there? It feels like your brain erased the mission. But the strange part is, your memory may be working normally.", "RL walks toward a bedroom doorway with a clear purposeful expression, one hand raised as if remembering an important task, clean home interior, strong visual storytelling."),
    ("image", "The first clue is the doorway itself. Your brain is constantly updating a mental model of where you are and what you are doing.", "RL crosses the doorway and the environment subtly changes from hallway to bedroom, visual metaphor of a mental context shift, readable composition."),
    ("video", "Crossing into a new place can act like an event boundary. Your attention starts processing the new scene, and the old intention can become less active.", "RL steps through the doorway, pauses, looks around as the camera gently pushes in; subtle environment transition suggests a new mental context."),
    ("image", "Imagine you were thinking, I need my charger. Then you enter the bedroom. Suddenly your brain is processing the bed, the desk, the objects around you, and whatever happens next.", "RL in a bedroom surrounded by simple visual cues: desk, phone charger, bed, backpack; the charger is present but not visually dominant, clean explainer composition."),
    ("video", "That does not mean the doorway magically deletes the memory. It is more like changing tabs. The information is still there, but another context becomes the one your attention is using.", "RL looks from the charger toward the new room as the camera performs a controlled rack-focus-like push; the original object remains in frame while attention shifts."),
    ("image", "And attention matters because remembering an intention is not passive. You have to keep that intention active while other information competes for attention.", "RL tries to remember the task while several simple everyday distractions appear around him, expressive face, uncluttered educational layout."),
    ("video", "So a tiny distraction can push the task behind the new information. RL stops, looks confused, and the camera makes a quick but controlled push toward his face.", "RL freezes with a confused expression, briefly scans the room, then looks directly toward camera; quick controlled push-in, expressive but natural movement."),
    ("image", "Here is the surprising part: the forgotten intention can return when you go back to the place or situation where you first had it.", "RL retraces his steps from the bedroom toward the doorway, expression changing from confused to recognition, clear before-and-after visual story."),
    ("video", "That is why retracing your steps sometimes works. You are rebuilding the context that was linked to the original intention.", "RL walks back through the doorway and suddenly remembers, small realization expression; camera pulls back to reveal the original context."),
    ("image", "Researchers describe this using ideas such as event boundaries and context changes. Your brain divides continuous experience into meaningful events so it can organize what happened.", "Clean visual explanation of RL moving through three simple connected spaces represented as distinct context panels, no text, strong geometric composition."),
    ("video", "Most of the time, that system is useful. The doorway is not the villain. It is simply one strong signal that the scene has changed.", "RL calmly walks through several everyday spaces while the camera tracks smoothly; each space changes cleanly without changing RL's appearance."),
    ("image", "So next time you walk into a room and forget why, do not assume your brain deleted the thought. You may have simply crossed into a new context. Retrace the situation and the missing intention may come back.", "RL stands in the doorway, smiles with recognition, then walks back toward the original task; satisfying final wide shot, warm clean lighting."),
]


def words(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text))


def run(cmd: list[str]) -> None:
    print("[V2 CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def make_tts(text: str, path: Path) -> float:
    """Use local Kokoro; fail loudly only if neither Kokoro nor espeak is available."""
    voice = os.environ.get("KOKORO_VOICE", "am_michael")
    speed = float(os.environ.get("KOKORO_SPEED", "1.05"))
    try:
        from kokoro import KPipeline
        kp = KPipeline(lang_code="a")
        chunks = []
        for _, _, audio in kp(text, voice=voice, speed=speed, split_pattern=r"\n+"):
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("Kokoro returned no audio")
        audio = np.concatenate(chunks)
        sf.write(path, audio, 24000)
        return len(audio) / 24000.0
    except Exception as exc:
        print(f"[V2 TTS] Kokoro unavailable: {exc}; using espeak-ng fallback.")
        run(["espeak-ng", "-s", "165", "-w", str(path), text])
        info = sf.info(path)
        return float(info.duration)


def still_clip(image: Path, audio_duration: float, out: Path, index: int) -> None:
    # Alternating push/pull gives still scenes visible movement without inventing new content.
    zoom = "zoompan=z='min(zoom+0.0007,1.08)':d=1:s=1280x720:fps=30" if index % 2 else "zoompan=z='if(lte(zoom,1.001),1.08,max(1.0,zoom-0.0007))':d=1:s=1280x720:fps=30"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-loop", "1", "-i", str(image),
         "-t", f"{audio_duration:.3f}", "-vf", zoom + ",format=yuv420p", "-r", "30",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-an", str(out)])


def fit_video(video: Path, duration: float, out: Path) -> None:
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stream_loop", "-1", "-i", str(video),
         "-t", f"{duration:.3f}", "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
         "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-an", str(out)])


def concat(parts: list[Path], out: Path) -> None:
    manifest = OUT / "concat.txt"
    manifest.write_text("\n".join(f"file '{p.as_posix()}'" for p in parts) + "\n", encoding="utf-8")
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest),
         "-c", "copy", str(out)])


def mux_audio(video: Path, audio_files: list[Path], out: Path) -> None:
    manifest = OUT / "audio_concat.txt"
    manifest.write_text("\n".join(f"file '{p.as_posix()}'" for p in audio_files) + "\n", encoding="utf-8")
    audio = OUT / "narration.wav"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c:a", "pcm_s16le", str(audio)])
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-i", str(audio),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)])


def main() -> None:
    topic = "Why do you forget why you walked into a room?"
    print(f"[V2] TEST ONLY: {topic}")
    print(f"[V2] scenes={len(SCENES)}, target animation={sum(k == 'video' for k, _, _ in SCENES)}/{len(SCENES)}")

    external_master = os.environ.get("RL_REFERENCE_IMAGE", "").strip()
    master = Path(external_master) if external_master else IMG / "rl_master_generated.png"
    if not master.exists():
        master_prompt = RL_PROMPT + " Full-body canonical character reference, neutral standing pose, plain light background, front three-quarter view, highly readable face and clothing. No text."
        generate_image(master_prompt, master)
    print(f"[V2] canonical RL reference: {master}")

    audio_files: list[Path] = []
    parts: list[Path] = []
    results = []
    total_words = 0
    animation_ok = 0
    for i, (kind, narration, visual) in enumerate(SCENES, 1):
        total_words += words(narration)
        scene_prompt = RL_PROMPT + "\n\nScene direction: " + visual
        image = IMG / f"scene_{i:02d}.png"
        audio = AUD / f"scene_{i:02d}.wav"
        if not image.exists():
            generate_image(scene_prompt, image, reference_path=master)
        duration = make_tts(narration, audio)
        audio_files.append(audio)

        clip = VID / f"scene_{i:02d}.mp4"
        animated = False
        if kind == "video":
            motion_prompt = visual + " Keep RL's identity and outfit fixed. Natural subtle character motion, clean educational animation, controlled camera movement, no text or subtitles."
            animated = animate_or_fallback(image, motion_prompt, clip, reference_path=master)
        if animated:
            animation_ok += 1
            results.append({"scene": i, "requested": "animation", "actual": "animation", "duration": round(duration, 2)})
            fit = VID / f"scene_{i:02d}_fit.mp4"
            fit_video(clip, duration, fit)
            parts.append(fit)
        else:
            clip = VID / f"scene_{i:02d}_still.mp4"
            still_clip(image, duration, clip, i)
            results.append({"scene": i, "requested": kind, "actual": "image", "duration": round(duration, 2)})
            parts.append(clip)

    silent = OUT / "v2_silent.mp4"
    final = OUT / "relic_loop_v2_room_memory_test.mp4"
    concat(parts, silent)
    mux_audio(silent, audio_files, final)
    manifest = {
        "topic": topic,
        "test_only": True,
        "scene_count": len(SCENES),
        "word_count": total_words,
        "requested_animation_ratio": round(sum(k == 'video' for k, _, _ in SCENES) / len(SCENES), 3),
        "successful_animation_ratio": round(animation_ok / len(SCENES), 3),
        "animation_circuit_breaker": "scene-level fallback to still image",
        "youtube_upload": False,
        "scenes": results,
        "output": str(final),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"[V2] COMPLETE: {final}")


if __name__ == "__main__":
    main()
