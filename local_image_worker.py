#!/usr/bin/env python3
"""CPU-only local image worker for Motive Unknown.

Runs in an isolated virtual environment so the local image stack cannot change
the main Kokoro/LLM dependency set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from optimum.intel import OVLatentConsistencyModelPipeline


LABELS = [
    "ERA / HISTORICAL CONTEXT",
    "SETTING",
    "CHARACTERS",
    "VISIBLE ACTION",
    "PROPS / SYMBOLS",
    "MOOD",
    "NARRATION BEAT",
    "SHOT DIRECTION",
    "CONTINUITY",
]


def extract_section(prompt: str, label: str) -> str:
    marker = f"{label}:"
    start = prompt.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    ends = [
        prompt.find(f"\n{other}:", start)
        for other in LABELS
        if other != label
    ]
    ends = [value for value in ends if value >= 0]
    end = min(ends) if ends else len(prompt)
    return " ".join(prompt[start:end].split())


def trim_words(value: str, count: int) -> str:
    return " ".join(value.split()[:count])


def compact_prompt(prompt: str, tokenizer) -> tuple[str, int]:
    """Create a useful prompt that stays safely below CLIP's 77-token limit."""
    era = extract_section(prompt, "ERA / HISTORICAL CONTEXT")
    setting = extract_section(prompt, "SETTING")
    chars = extract_section(prompt, "CHARACTERS")
    action = extract_section(prompt, "VISIBLE ACTION")
    props = extract_section(prompt, "PROPS / SYMBOLS")
    mood = extract_section(prompt, "MOOD")
    shot = extract_section(prompt, "SHOT DIRECTION")

    style = (
        "Polished 2D historical cartoon, cinematic storybook, expressive believable "
        "people, richly layered environment, crisp ink contours, painterly cel-shaded "
        "color, warm natural light."
    )
    negative = (
        "No readable text, logos, modern objects, photorealism, 3D CGI, anime, "
        "stick figures, doodles, cars, asphalt or lane markings."
    )

    candidates = [
        " ".join(
            part for part in [
                style,
                f"Era: {trim_words(era, 7)}." if era else "",
                f"Setting: {trim_words(setting, 9)}." if setting else "",
                f"People: {trim_words(chars, 8)}." if chars else "",
                f"Action: {trim_words(action, 10)}." if action else "",
                f"Props: {trim_words(props, 6)}." if props else "",
                f"Mood: {trim_words(mood, 3)}." if mood else "",
                f"Shot: {trim_words(shot, 7)}." if shot else "",
                negative,
            ] if part
        ),
        " ".join(
            part for part in [
                style,
                f"Era: {trim_words(era, 7)}." if era else "",
                f"Setting: {trim_words(setting, 10)}." if setting else "",
                f"People: {trim_words(chars, 8)}." if chars else "",
                f"Action: {trim_words(action, 12)}." if action else "",
                f"Props: {trim_words(props, 6)}." if props else "",
                negative,
            ] if part
        ),
        " ".join(
            part for part in [
                style,
                f"Era: {trim_words(era, 7)}." if era else "",
                f"Setting: {trim_words(setting, 12)}." if setting else "",
                f"People: {trim_words(chars, 8)}." if chars else "",
                f"Action: {trim_words(action, 14)}." if action else "",
                negative,
            ] if part
        ),
        " ".join(
            part for part in [
                style,
                f"Setting: {trim_words(setting, 14)}." if setting else "",
                f"Action: {trim_words(action, 16)}." if action else "",
                negative,
            ] if part
        ),
    ]

    for candidate in candidates:
        count = len(tokenizer(candidate, add_special_tokens=True)["input_ids"])
        if count <= 75:
            return candidate, count

    candidate = candidates[-1]
    words = candidate.split()
    while len(tokenizer(candidate, add_special_tokens=True)["input_ids"]) > 75 and len(words) > 24:
        # Keep style and safety negatives; trim from the positive scene description.
        del words[len(style.split()):len(style.split()) + 1]
        candidate = " ".join(words)
    count = len(tokenizer(candidate, add_special_tokens=True)["input_ids"])
    return candidate, count


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: local_image_worker.py REQUEST_JSON")

    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding="utf-8"))

    model_id = str(request["model_id"])
    prompt = str(request["prompt"])
    output_path = Path(request["output_path"])
    seed = int(request["seed"])
    width = int(request.get("width", 768))
    height = int(request.get("height", 512))
    steps = int(request.get("steps", 4))

    print(f"[LOCAL WORKER] model={model_id} seed={seed} size={width}x{height} steps={steps}")
    pipeline = OVLatentConsistencyModelPipeline.from_pretrained(
        model_id,
        safety_checker=None,
    )
    pipeline.set_progress_bar_config(disable=True)

    compact, token_count = compact_prompt(prompt, pipeline.tokenizer)
    print(f"[LOCAL WORKER] prompt_tokens={token_count}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipeline(
        compact,
        num_inference_steps=steps,
        guidance_scale=8.0,
        lcm_origin_steps=50,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = ImageOps.fit(
        image.convert("RGB"),
        (1024, 576),
        method=Image.Resampling.LANCZOS,
    )
    image.save(output_path, format="JPEG", quality=92, optimize=True)

    if not output_path.exists() or output_path.stat().st_size < 10000:
        raise RuntimeError("Local worker produced no valid image.")

    print(f"[LOCAL WORKER] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
