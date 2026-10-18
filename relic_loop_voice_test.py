"""Generate standalone Relic Loop narrator samples without rendering a video."""
from __future__ import annotations

import asyncio
from pathlib import Path
import edge_tts

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
    "roger": "en-US-RogerNeural",
    "guy": "en-US-GuyNeural",
    "brian": "en-US-BrianNeural",
}

OUT = Path("voice_samples")
OUT.mkdir(parents=True, exist_ok=True)


async def synthesize(name: str, voice: str) -> None:
    target = OUT / f"{name}_{voice}.mp3"
    print(f"[VOICE TEST] {voice} -> {target}")
    communicate = edge_tts.Communicate(
        SAMPLE_TEXT,
        voice,
        rate="+6%",
        volume="+0%",
        pitch="+0Hz",
    )
    await communicate.save(str(target))
    if not target.exists() or target.stat().st_size < 1000:
        raise RuntimeError(f"No usable audio produced for {voice}")


async def main() -> None:
    for name, voice in VOICES.items():
        await synthesize(name, voice)
    (OUT / "README.txt").write_text(
        "Relic Loop voice test\n\n"
        "All three samples use the exact same narration and a slightly energetic +6% rate.\n"
        "Roger = lively, Guy = passion, Brian = approachable/casual/sincere.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
