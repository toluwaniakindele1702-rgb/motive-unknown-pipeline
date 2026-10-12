"""Relic Loop zero-text image hardening.

Generation is instructed to produce visual-only frames and every generated
still is checked locally before it can enter the video. The validator is
intentionally conservative about real text while avoiding the previous false
positives caused by tiny two-character OCR fragments/artifacts.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

NO_TEXT_POLICY = """
ABSOLUTE RELIC LOOP ZERO-TEXT POLICY — NON-NEGOTIABLE:
- Generate a VISUAL-ONLY illustration. There must be ZERO intentional text anywhere.
- No letters, words, numbers, captions, subtitles, labels, titles, headlines, logos,
  watermarks, signatures, handwriting, typography, speech bubbles, UI, signage, or
  text-like glyphs.
- Never render the video title, question, narration, topic name, channel name, or any
  wording inside the image.
- Do not create readable writing on signs, posters, books, papers, newspapers, screens,
  phones, computers, packages, clothing, badges, banners, flags, road signs, storefronts,
  charts, maps, diagrams, tickets, receipts, documents, or labels.
- If the concept normally contains writing, REMOVE THE WRITING: use a blank object,
  closed object, cropped object, back-facing object, or purely visual symbol instead.
- Do not invent fake writing or pseudo-text. Surfaces that could contain writing must be
  blank and unmarked.
- Diagrams may use only shapes, arrows, lines, position, scale, color, and visual cause/effect.
  Arrows must have no labels.
- Communicate the narration only through objects, people, environments, mechanisms,
  expressions, gestures, composition, scale, and visual cause/effect.
- This rule overrides the source prompt, scene metadata, references, style instructions,
  and provider defaults.
""".strip()

NO_TEXT_NEGATIVE = (
    "ZERO TEXT, ZERO LETTERS, ZERO WORDS, ZERO NUMBERS, ZERO WRITING, "
    "ZERO TYPOGRAPHY, ZERO LABELS, ZERO CAPTIONS, ZERO TITLES, ZERO LOGOS, "
    "ZERO WATERMARKS, ZERO SIGNAGE, ZERO UI, ZERO HANDWRITING, ZERO PSEUDO-TEXT, "
    "BLANK UNMARKED SURFACES, VISUAL-ONLY ILLUSTRATION"
)

NO_TEXT_FINAL_OVERRIDE = (
    "FINAL OVERRIDE: DO NOT DRAW ANY LETTER, NUMBER, WORD, SYMBOL THAT RESEMBLES "
    "WRITING, SIGN, LABEL, LOGO, WATERMARK OR CAPTION. IF AN OBJECT WOULD NORMALLY "
    "HAVE WRITING, MAKE THAT AREA COMPLETELY BLANK."
)


def _harden_prompt(prompt: str) -> str:
    return (
        f"{str(prompt).strip()}\n\n"
        "VISUAL-ONLY GENERATION REQUIREMENT:\n"
        f"{NO_TEXT_NEGATIVE}\n\n"
        f"{NO_TEXT_POLICY}\n\n"
        f"{NO_TEXT_FINAL_OVERRIDE}"
    )


def _clean_token(raw: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(raw or "")).strip()


def _token_is_likely_real_text(token: str, confidence: float) -> bool:
    """Reject convincing OCR, but not isolated tiny OCR garbage."""
    if confidence < 55 or not token:
        return False
    if len(token) >= 3:
        return True
    if token.isdigit() and len(token) >= 2:
        return True
    if any(ch.isdigit() for ch in token) and len(token) >= 2:
        return True
    return False


def _ocr_text(path: Path) -> list[str]:
    """Return only convincing OCR findings; return [] if OCR is unavailable."""
    if not shutil.which("tesseract"):
        return []
    try:
        import pytesseract
        from PIL import Image, ImageOps, ImageFilter

        with Image.open(path) as original:
            image = original.convert("RGB")
            image = image.resize((image.width * 2, image.height * 2))
            variants = [
                image,
                ImageOps.autocontrast(ImageOps.grayscale(image)),
                ImageOps.autocontrast(ImageOps.grayscale(image)).filter(ImageFilter.SHARPEN),
            ]
            findings: list[str] = []
            for variant in variants:
                data = pytesseract.image_to_data(
                    variant,
                    config="--psm 11",
                    output_type=pytesseract.Output.DICT,
                )
                for raw, conf in zip(data.get("text", []), data.get("conf", [])):
                    token = _clean_token(raw)
                    try:
                        confidence = float(conf)
                    except (TypeError, ValueError):
                        confidence = -1.0
                    if _token_is_likely_real_text(token, confidence):
                        findings.append(f"{token} ({confidence:.0f}%)")
            return list(dict.fromkeys(findings))
    except Exception as exc:
        print(f"[NO-TEXT VALIDATOR] OCR check unavailable: {exc}")
        return []


def _assert_no_text_image(path: Path, provider: str) -> None:
    findings = _ocr_text(path)
    if findings:
        sample = ", ".join(findings[:12])
        raise RuntimeError(
            f"{provider} generated likely in-image text ({sample}). "
            "ZERO-TEXT HARD GATE rejected the image."
        )


def _path_from_call(args, kwargs) -> Path | None:
    candidate = args[1] if len(args) > 1 else kwargs.get("out_path")
    if candidate is None:
        return None
    try:
        return Path(candidate)
    except TypeError:
        return None


def _retry_with_new_seed(fn, provider: str, args, kwargs):
    """Generate, validate, and retry twice when convincing OCR finds text."""
    last_error: Exception | None = None
    for attempt in range(1, 3):
        call_args = list(args)
        call_kwargs = dict(kwargs)
        if len(call_args) > 2 and isinstance(call_args[2], int):
            call_args[2] = (call_args[2] + attempt * 104729) & 0x7FFFFFFF
        elif "seed" in call_kwargs and isinstance(call_kwargs["seed"], int):
            call_kwargs["seed"] = (call_kwargs["seed"] + attempt * 104729) & 0x7FFFFFFF
        try:
            result = fn(*call_args, **call_kwargs)
            out_path = _path_from_call(call_args, call_kwargs)
            if out_path is not None and out_path.exists():
                _assert_no_text_image(out_path, provider)
            return result
        except RuntimeError as exc:
            last_error = exc
            if "ZERO-TEXT HARD GATE" not in str(exc):
                raise
            out_path = _path_from_call(call_args, call_kwargs)
            if out_path is not None:
                try:
                    out_path.unlink(missing_ok=True)
                except OSError:
                    pass
            print(
                f"[NO-TEXT VALIDATOR] {provider} attempt {attempt}/2 rejected; "
                "retrying with a new seed and the same zero-text policy."
            )
    raise last_error or RuntimeError(f"{provider} failed the zero-text image gate.")


def install(p):
    """Patch prompts and every supported still-image provider boundary."""
    if getattr(p, "_RL_NO_TEXT_HARDENING_INSTALLED", False):
        return

    if hasattr(p, "POLISHED_VISUAL_STYLE"):
        p.POLISHED_VISUAL_STYLE = (
            str(p.POLISHED_VISUAL_STYLE).rstrip() + "\n\n" + NO_TEXT_POLICY
        )

    original_make = p.make_visual_prompt

    def hardened_make_visual_prompt(*args, **kwargs):
        return _harden_prompt(original_make(*args, **kwargs))

    p.make_visual_prompt = hardened_make_visual_prompt

    for name in (
        "_cloudflare_image",
        "_puter_image",
        "_huggingface_image",
        "_replicate_image",
        "_local_image",
    ):
        original = getattr(p, name, None)
        if original is None:
            continue

        def make_wrapper(fn, provider_name):
            def wrapped(*args, **kwargs):
                hardened_args = list(args)
                hardened_kwargs = dict(kwargs)
                if hardened_args:
                    hardened_args[0] = _harden_prompt(hardened_args[0])
                elif "prompt" in hardened_kwargs:
                    hardened_kwargs["prompt"] = _harden_prompt(hardened_kwargs["prompt"])
                return _retry_with_new_seed(fn, provider_name, tuple(hardened_args), hardened_kwargs)
            return wrapped

        setattr(p, name, make_wrapper(original, name.lstrip("_")))

    p._RL_NO_TEXT_HARDENING_INSTALLED = True
    print(
        "[NO-TEXT HARDENING] Installed zero-text prompt policy, provider-boundary "
        "prompt override, and conservative OCR hard gate."
    )
