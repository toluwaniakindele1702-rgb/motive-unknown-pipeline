"""Absolute no-text image hardening for Relic Loop production.

Every generated scene image must be visually explainable without any in-image
writing. This module patches the final visual prompt and every image-provider
entry point so a provider cannot accidentally receive the old text-permitting
prompt.
"""
from __future__ import annotations

NO_TEXT_POLICY = """
ABSOLUTE RELIC LOOP NO-TEXT IMAGE POLICY — NON-NEGOTIABLE:
- The generated image must contain ZERO text of any kind.
- ZERO readable text, letters, words, numbers, captions, subtitles, labels, titles, headlines, logos, watermarks, signatures, speech bubbles, UI text, fake handwriting, pseudo-writing, glyphs, symbols that resemble writing, or decorative lettering.
- Never place the video title, central question, narration, or any other wording inside the image.
- Signs, posters, books, documents, newspapers, screens, phone displays, computer displays, product packages, clothing, badges, banners, flags, road signs, storefronts, and charts must be blank/unmarked or shown from an angle where no writing is visible.
- If the narration mentions a written object, depict the physical object as blank/closed/unreadable and communicate the idea through the object, action, composition, arrows, shapes, or visual cause-and-effect — never through words.
- Use purely visual storytelling: objects, people when required, animals, environments, mechanisms, diagrams without labels, arrows without labels, cutaways, comparisons, scale, motion, gestures, and expressions.
- This rule overrides any other instruction or reference image that suggests adding text.
- If the image model is uncertain, choose a completely blank surface rather than inventing writing.
""".strip()


def _harden_prompt(prompt: str) -> str:
    return f"{str(prompt).strip()}\n\n{NO_TEXT_POLICY}"


def install(p):
    """Patch Relic Loop's prompt and provider boundaries once per process."""
    if getattr(p, "_RL_NO_TEXT_HARDENING_INSTALLED", False):
        return

    original_make = p.make_visual_prompt

    def hardened_make_visual_prompt(*args, **kwargs):
        return _harden_prompt(original_make(*args, **kwargs))

    p.make_visual_prompt = hardened_make_visual_prompt

    # Patch every provider boundary too. This catches production renders and
    # direct visual-test paths that bypass make_visual_prompt().
    for name in ("_cloudflare_image", "_puter_image", "_huggingface_image", "_replicate_image", "_local_image"):
        original = getattr(p, name, None)
        if original is None:
            continue

        def make_wrapper(fn):
            def wrapped(*args, **kwargs):
                args = list(args)
                if args:
                    args[0] = _harden_prompt(args[0])
                elif "prompt" in kwargs:
                    kwargs["prompt"] = _harden_prompt(kwargs["prompt"])
                return fn(*args, **kwargs)
            return wrapped

        setattr(p, name, make_wrapper(original))

    p._RL_NO_TEXT_HARDENING_INSTALLED = True
    print("[NO-TEXT HARDENING] Installed absolute zero-text policy on visual prompts and image providers.")
