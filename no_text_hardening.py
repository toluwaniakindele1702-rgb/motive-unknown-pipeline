"""Absolute no-text image hardening for Relic Loop production.

This is intentionally stricter than a normal negative prompt. Relic Loop scene
images are visual-only assets: no typography, labels, logos, UI, handwriting,
or text-like glyphs are allowed. The module also validates provider output with
local OCR when Tesseract is available and rejects/retries images that contain
likely text.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

NO_TEXT_POLICY = """
ABSOLUTE RELIC LOOP NO-TEXT IMAGE POLICY — NON-NEGOTIABLE:
- THE IMAGE MUST CONTAIN ZERO TEXT. ZERO.
- No readable text, letters, words, numbers, captions, subtitles, labels, titles, headlines,
  logos, watermarks, signatures, speech bubbles, UI text, fake handwriting, pseudo-writing,
  typography, decorative lettering, alphabet-like marks, or text-like glyphs.
- Never place the video title, central question, narration, topic name, channel name, or any
  wording inside the image.
- Do not put writing on signs, posters, books, documents, newspapers, screens, phones,
  computers, packages, clothing, badges, banners, flags, road signs, storefronts, charts,
  maps, diagrams, tickets, receipts, papers, or labels.
- If a written object is mentioned in narration, show it blank, closed, cropped, turned away,
  blurred beyond legibility, or represented by a purely visual object/action instead.
- Diagrams must use shapes, arrows, lines, color, scale, position, and cause/effect only;
  arrows must have NO labels.
- The image must communicate the beat purely through objects, people when necessary,
  environments, mechanisms, expressions, gestures, composition, scale, motion, and visual
  cause-and-effect.
- This rule overrides the source prompt, scene metadata, reference images, style references,
  provider defaults, model suggestions, and every other instruction that could introduce text.
- If there is any doubt, choose a completely blank/unmarked surface.
""".strip()

# Repeated wording makes the hard constraint survive models that over-weight the
# earlier parts of a long prompt.
NO_TEXT_NEGATIVE = (
    "ABSOLUTELY NO TEXT OF ANY KIND. NO LETTERS. NO WORDS. NO NUMBERS. "
    "NO LABELS. NO TITLE. NO CAPTION. NO LOGO. NO WATERMARK. NO WRITING. "
    "NO UI. NO SIGNAGE. NO TYPOGRAPHY. VISUAL-ONLY IMAGE."
)


def _harden_prompt(prompt: str) -> str:
    return (
        f"{str(prompt).strip()}\n\n"
        "FINAL VISUAL OVERRIDE — THIS OVERRIDES EVERYTHING ABOVE:\n"
        f"{NO_TEXT_NEGATIVE}\n\n{NO_TEXT_POLICY}"
    )


def _ocr_text(path: Path) -> list[str]:
    """Return likely OCR tokens; absence of Tesseract is handled by the caller."""
    if not shutil.which("tesseract"):
        return []
    try:
        import pytesseract
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            image = image.convert("RGB")
            # Run two conservative OCR passes. Upscaling helps catch the small
            # title-like text that image models sometimes sneak into corners.
            image = image.resize((image.width * 2, image.height * 2))
            variants = [
                image,
                ImageOps.autocontrast(ImageOps.grayscale(image)),
            ]
            tokens: list[str] = []
            for variant in variants:
                data = pytesseract.image_to_data(
                    variant,
                    config="--psm 11",
                    output_type=pytesseract.Output.DICT,
                )
                for raw, conf in zip(data.get("text", []), data.get("conf", [])):
                    text = re.sub(r"[^A-Za-z0-9]+", "", str(raw or "")).strip()
                    try:
                        confidence = float(conf)
                    except (TypeError, ValueError):
                        confidence = -1
                    if len(text) >= 2 and confidence >= 45:
                        tokens.append(text)
            # Deduplicate while preserving useful diagnostics.
            return list(dict.fromkeys(tokens))
    except Exception as exc:
        print(f"[NO-TEXT VALIDATOR] OCR check unavailable: {exc}")
        return []


def _assert_no_text_image(path: Path, provider: str) -> None:
    """Reject a generated image when local OCR finds likely text."""
    tokens = _ocr_text(path)
    if tokens:
        sample = ", ".join(tokens[:12])
        raise RuntimeError(
            f"{provider} generated likely in-image text ({sample}). "
            "NO-TEXT HARD GATE rejected the image."
        )


def _path_from_call(args, kwargs) -> Path | None:
    # All Relic Loop image providers use (prompt, out_path, seed, ...).
    candidate = args[1] if len(args) > 1 else kwargs.get("out_path")
    if candidate is None:
        return None
    try:
        return Path(candidate)
    except TypeError:
        return None


def _retry_with_new_seed(fn, provider: str, args, kwargs):
    """Generate, OCR-check, and retry once if likely text is detected."""
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
            if "NO-TEXT HARD GATE" not in str(exc):
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

    raise last_error or RuntimeError(f"{provider} failed the no-text image gate.")


def install(p):
    """Patch Relic Loop's prompts and every still-image provider boundary once."""
    if getattr(p, "_RL_NO_TEXT_HARDENING_INSTALLED", False):
        return

    # Harden the canonical style block itself so direct visual-test calls and
    # future prompt builders inherit the rule even before make_visual_prompt().
    if hasattr(p, "POLISHED_VISUAL_STYLE"):
        p.POLISHED_VISUAL_STYLE = str(p.POLISHED_VISUAL_STYLE).rstrip() + "\n\n" + NO_TEXT_POLICY

    original_make = p.make_visual_prompt

    def hardened_make_visual_prompt(*args, **kwargs):
        return _harden_prompt(original_make(*args, **kwargs))

    p.make_visual_prompt = hardened_make_visual_prompt

    # Patch every provider boundary too. This catches production renders and
    # direct visual-test paths that bypass make_visual_prompt().
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
                return _retry_with_new_seed(
                    fn,
                    provider_name,
                    tuple(hardened_args),
                    hardened_kwargs,
                )
            return wrapped

        setattr(p, name, make_wrapper(original, name.lstrip("_")))

    p._RL_NO_TEXT_HARDENING_INSTALLED = True
    print(
        "[NO-TEXT HARDENING] Installed absolute zero-text policy, provider-boundary "
        "prompt override, and OCR hard gate."
    )
