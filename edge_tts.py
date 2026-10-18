"""Small compatibility shim: keep the assembler API, but synthesize with Kokoro Liam.

The production assembler historically imported edge_tts.Communicate. Rather than
rewrite the timing/assembly code, this module provides that tiny interface while
routing every narration request to Kokoro's American-English Liam voice.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

__version__ = "kokoro-liam-compat"
VOICE_ID = os.getenv("RELIC_LOOP_KOKORO_VOICE", "am_liam")
SAMPLE_RATE = 24000

_pipeline: KPipeline | None = None


def _get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        print("[KOKORO] Loading American-English pipeline...")
        _pipeline = KPipeline(lang_code="a")
        print("[KOKORO] Pipeline loaded.")
    return _pipeline


def _synthesize(text: str) -> np.ndarray:
    pipeline = _get_pipeline()
    chunks: list[np.ndarray] = []
    for _, _, audio in pipeline(text, voice=VOICE_ID, speed=1.0):
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size:
            chunks.append(arr)
    if not chunks:
        raise RuntimeError("Kokoro returned no audio chunks")
    return np.concatenate(chunks)


class Communicate:
    """Minimal edge_tts.Communicate-compatible wrapper used by the assembler."""

    def __init__(self, text: str, voice: str, rate: str = "+0%", volume: str = "+0%", pitch: str = "+0Hz") -> None:
        self.text = text
        self.requested_voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch

    async def save(self, output_path: str) -> None:
        if not self.text.strip():
            raise ValueError("Cannot synthesize empty narration")
        print(f"[KOKORO] Voice={VOICE_ID}; requested_voice={self.requested_voice}; generating narration")
        audio = await asyncio.to_thread(_synthesize, self.text)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="kokoro_") as td:
            wav = Path(td) / "narration.wav"
            sf.write(wav, audio, SAMPLE_RATE, subtype="PCM_16")
            if target.suffix.lower() == ".wav":
                target.write_bytes(wav.read_bytes())
            else:
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "192k", str(target)],
                    check=True,
                )
        if not target.exists() or target.stat().st_size < 1000:
            raise RuntimeError(f"Kokoro produced no usable audio at {target}")
        print(f"[KOKORO] Wrote {target} ({target.stat().st_size} bytes)")
