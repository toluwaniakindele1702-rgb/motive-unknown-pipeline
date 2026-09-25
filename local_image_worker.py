#!/usr/bin/env python3
"""CPU-only local image worker for Motive Unknown.

Runs in an isolated virtual environment so the local image stack cannot change
the main Kokoro/LLM dependency set.
"""

from __future__ import annotations

import json
import os
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
    "VISUAL DEVICE",
    "SHOT DIRECTION",
    "CONTINUITY",
]


PIPELINE = None
LOCAL_MODEL_ID = os.environ.get(
    "MOTIVE_LOCAL_IMAGE_MODEL",
    "OpenVINO/LCM_Dreamshaper_v7-int8-ov",
).strip()


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
    """Preserve the exact narration beat while staying below CLIP's 77-token limit."""
    era = extract_section(prompt, "ERA / HISTORICAL CONTEXT")
    setting = extract_section(prompt, "SETTING")
    chars = extract_section(prompt, "CHARACTERS")
    action = extract_section(prompt, "VISIBLE ACTION")
    props = extract_section(prompt, "PROPS / SYMBOLS")
    mood = extract_section(prompt, "MOOD")
    beat = extract_section(prompt, "NARRATION BEAT")
    device = extract_section(prompt, "VISUAL DEVICE")
    shot = extract_section(prompt, "SHOT DIRECTION")

    style = "Cinematic 2D historical illustration, premium storybook, believable people, painterly cel shading, crisp ink, warm natural light."
    negative = "No readable text, logos, modern objects, photorealism, CGI, anime, cars, asphalt, lane markings."

    fields = [
        style,
        f"BEAT: {trim_words(beat, 18)}." if beat else "",
        f"ACTION: {trim_words(action, 10)}." if action else "",
        f"SETTING: {trim_words(setting, 8)}." if setting else "",
        f"DEVICE: {trim_words(device, 9)}." if device else "",
        f"PEOPLE: {trim_words(chars, 7)}." if chars else "",
        f"PROPS: {trim_words(props, 5)}." if props else "",
        f"ERA: {trim_words(era, 5)}." if era else "",
        f"MOOD: {trim_words(mood, 3)}." if mood else "",
        f"SHOT: {trim_words(shot, 6)}." if shot else "",
        negative,
    ]

    candidate_parts = [fields[0]]
    for field in fields[1:]:
        if not field:
            continue
        trial = " ".join(candidate_parts + [field])
        if len(tokenizer(trial, add_special_tokens=True)["input_ids"]) <= 75:
            candidate_parts.append(field)

    candidate = " ".join(candidate_parts)

    if len(tokenizer(candidate, add_special_tokens=True)["input_ids"]) > 75:
        candidate = " ".join(
            x for x in [
                style,
                f"BEAT: {trim_words(beat, 24)}." if beat else "",
                f"ACTION: {trim_words(action, 12)}." if action else "",
                negative,
            ] if x
        )
        words = candidate.split()
        while len(tokenizer(candidate, add_special_tokens=True)["input_ids"]) > 75 and len(words) > 20:
            words.pop(max(1, len(words) // 2))
            candidate = " ".join(words)

    count = len(tokenizer(candidate, add_special_tokens=True)["input_ids"])
    return candidate, count


def generate_from_request(request: dict) -> None:
    model_id = str(request["model_id"])
    prompt = str(request["prompt"])
    output_path = Path(request["output_path"])
    seed = int(request["seed"])
    width = int(request.get("width", 768))
    height = int(request.get("height", 512))
    steps = int(request.get("steps", 4))

    print(
        "[LOCAL WORKER] generate seed={} size={}x{} steps={}".format(
            seed, width, height, steps
        ),
        flush=True,
    )

    compact, token_count = compact_prompt(prompt, PIPELINE.tokenizer)
    print("[LOCAL WORKER] prompt_tokens={}".format(token_count), flush=True)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = PIPELINE(
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

    print("[LOCAL WORKER] wrote {}".format(output_path), flush=True)


def run_server() -> int:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("cmd") == "shutdown":
            print("__LOCAL_SHUTDOWN__", flush=True)
            return 0
        if request.get("cmd") != "generate":
            print(
                "__LOCAL_ERROR__ unknown command: {}".format(request.get("cmd")),
                flush=True,
            )
            continue
        try:
            generate_from_request(request)
            print(
                "__LOCAL_OK__ {}".format(request.get("output_path", "")),
                flush=True,
            )
        except Exception as exc:
            print(
                "__LOCAL_ERROR__ {}: {}".format(type(exc).__name__, exc),
                flush=True,
            )
    return 0


def main() -> int:
    global PIPELINE

    if len(sys.argv) == 2 and sys.argv[1] == "--server":
        print("[LOCAL WORKER] Loading model once: {}".format(LOCAL_MODEL_ID), flush=True)
        PIPELINE = OVLatentConsistencyModelPipeline.from_pretrained(
            LOCAL_MODEL_ID,
            safety_checker=None,
        )
        PIPELINE.set_progress_bar_config(disable=True)
        print("[LOCAL WORKER] Model ready.", flush=True)
        return run_server()

    if len(sys.argv) != 2:
        raise SystemExit("Usage: local_image_worker.py REQUEST_JSON | --server")

    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding="utf-8"))

    PIPELINE = OVLatentConsistencyModelPipeline.from_pretrained(
        str(request["model_id"]),
        safety_checker=None,
    )
    PIPELINE.set_progress_bar_config(disable=True)
    generate_from_request(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
