"""Generate standalone Relic Loop narrator samples without rendering a video."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_TEXT = (
    "He was five seconds late, so he made the worst decision of his morning. "
    "He skipped breakfast, refused to tie his shoe, and sprinted for the bus. "
    "He actually caught it — then realized it was going in the opposite direction. "
    "Instead of getting off, he decided to stay on because he was already embarrassed. "
    "Twenty minutes later, the driver stopped at the airport. "
    "He stared out the window, looked at his untied shoe, and finally admitted one thing: "
    "saving five seconds had somehow cost him an entire morning."
)

VOICES = {
    "michael": "am_michael",
    "puck": "am_puck",
    "liam": "am_liam",
}

OUT = Path("voice_samples")
OUT.mkdir(parents=True, exist_ok=True)


def synthesize(name: str, voice: str, pipeline: KPipeline) -> None:
    target = OUT / f"kokoro_{name}.wav"
    print(f"[KOKORO VOICE TEST] {voice} -> {target}")
    chunks = []
    for _, _, audio in pipeline(SAMPLE_TEXT, voice=voice, speed=1.06):
        chunks.append(audio.numpy())
    if not chunks:
        raise RuntimeError(f"No audio produced for {voice}")
    audio = np.concatenate(chunks)
    sf.write(str(target), audio, 24000)
    if not target.exists() or target.stat().st_size < 1000:
        raise RuntimeError(f"No usable audio produced for {voice}")


def main() -> None:
    pipeline = KPipeline(lang_code="a")
    for name, voice in VOICES.items():
        synthesize(name, voice, pipeline)
    (OUT / "README.txt").write_text(
        "Relic Loop Kokoro voice test\n\n"
        "All samples use the exact same narration and speed 1.06.\n"
        "Michael = warm/trustworthy, Puck = playful/energetic, Liam = friendly/conversational.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
