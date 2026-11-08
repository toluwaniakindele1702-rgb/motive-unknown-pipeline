"""Raphael browser UI adapter for Chispa-Pifia.

Uses the public Raphael web interface, not private endpoints. It does not solve,
bypass, or automate around CAPTCHA/Turnstile. If the UI requires human verification
or sign-in and no valid browser session is supplied, the provider fails cleanly and
the configured Cloudflare fallback is used.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from io import BytesIO

from PIL import Image
from playwright.sync_api import sync_playwright

RAPHAEL_URL = os.getenv("RAPHAEL_URL", "https://raphael.app/ai-image-generator")
MODEL_PREFERENCE = [
    x.strip() for x in os.getenv(
        "RAPHAEL_MODEL_PREFERENCE",
        "GPT Image 2,Seedream 5.0 Pro,Nano Banana Pro,Seedream 4.5,Seedream 3.5"
    ).split(",") if x.strip()
]
STORAGE_STATE = os.getenv("RAPHAEL_STORAGE_STATE", "").strip()
STORAGE_STATE_JSON = os.getenv("RAPHAEL_STORAGE_STATE_JSON", "").strip()
if STORAGE_STATE_JSON and not STORAGE_STATE:
    # Optional user-supplied Playwright session state; never attempt to create
    # or bypass an account verification challenge.
    state_path = Path("/tmp/raphael_storage_state.json")
    try:
        json.loads(STORAGE_STATE_JSON)
        state_path.write_text(STORAGE_STATE_JSON, encoding="utf-8")
        STORAGE_STATE = str(state_path)
    except (ValueError, OSError) as exc:
        print(f"[RAPHAEL] ignoring invalid storage-state secret: {type(exc).__name__}")


def _visible(locator) -> bool:
    try:
        return locator.count() > 0 and locator.first.is_visible()
    except Exception:
        return False


def _select_model(page) -> str:
    # Prefer the best available quality model in this order. UI wording may evolve.
    for name in MODEL_PREFERENCE:
        candidates = page.get_by_text(re.compile(re.escape(name), re.I))
        for i in range(min(candidates.count(), 8)):
            item = candidates.nth(i)
            try:
                if item.is_visible():
                    item.click(timeout=2500)
                    time.sleep(1)
                    return name
            except Exception:
                continue
    return "Raphael default model (UI did not expose a matching model selector)"


def generate(prompt: str, refs: list[tuple[str, bytes, str]] | None = None) -> bytes:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context_args = {"viewport": {"width": 1440, "height": 1200}, "accept_downloads": True}
        if STORAGE_STATE and Path(STORAGE_STATE).is_file():
            context_args["storage_state"] = STORAGE_STATE
        context = browser.new_context(**context_args)
        page = context.new_page()
        try:
            page.goto(RAPHAEL_URL, wait_until="domcontentloaded", timeout=90000)
            page.wait_for_timeout(5000)

            body = (page.locator("body").inner_text(timeout=15000) or "").lower()
            if any(term in body for term in ("verify you are human", "turnstile", "captcha")):
                raise RuntimeError("Raphael requires human verification; automated verification is intentionally not attempted.")
            if any(term in body for term in ("sign in to continue", "sign in required", "log in to generate")):
                raise RuntimeError("Raphael requires a signed-in browser session. Configure RAPHAEL_STORAGE_STATE as a valid Playwright storage-state file.")

            # Attach canonical visual references when the UI exposes an upload control.
            uploads = page.locator('input[type="file"]')
            if refs and uploads.count():
                paths = []
                for idx, (name, blob, mime) in enumerate(refs[:4]):
                    suffix = ".png" if "png" in mime else ".jpg"
                    target = Path("/tmp") / f"raphael_reference_{idx}{suffix}"
                    target.write_bytes(blob)
                    paths.append(str(target))
                try:
                    uploads.first.set_input_files(paths, timeout=10000)
                    page.wait_for_timeout(1500)
                except Exception as exc:
                    print(f"[RAPHAEL] reference upload unavailable; continuing prompt-only: {exc}")

            model_used = _select_model(page)
            print(f"[RAPHAEL] selected model: {model_used}")

            prompt_boxes = page.locator("textarea, [contenteditable='true'], input[type='text']")
            prompt_box = None
            for i in range(prompt_boxes.count()):
                item = prompt_boxes.nth(i)
                try:
                    if item.is_visible() and item.is_enabled():
                        prompt_box = item
                        break
                except Exception:
                    pass
            if prompt_box is None:
                raise RuntimeError("Could not locate Raphael's prompt field; UI may have changed.")
            prompt_box.fill(prompt, timeout=15000)

            buttons = page.get_by_role("button")
            generate_button = None
            for label in ("Generate", "Create image", "Create"):
                candidate = page.get_by_role("button", name=re.compile(label, re.I))
                if _visible(candidate):
                    generate_button = candidate.first
                    break
            if generate_button is None:
                raise RuntimeError("Could not locate Raphael's Generate button; UI may have changed.")
            generate_button.click(timeout=15000)

            deadline = time.time() + int(os.getenv("RAPHAEL_TIMEOUT_SECONDS", "240"))
            last_error = ""
            while time.time() < deadline:
                page.wait_for_timeout(2500)
                text = (page.locator("body").inner_text(timeout=10000) or "").lower()
                if any(term in text for term in ("verify you are human", "turnstile", "captcha")):
                    raise RuntimeError("Raphael requested human verification; stopping without bypass.")
                if any(term in text for term in ("insufficient credits", "not enough credits", "sign in to generate")):
                    raise RuntimeError("Raphael account has no usable generation credit or requires sign-in.")
                imgs = page.locator("img")
                for idx in reversed(range(imgs.count())):
                    img = imgs.nth(idx)
                    try:
                        if not img.is_visible():
                            continue
                        dimensions = img.evaluate("(e) => ({w:e.naturalWidth,h:e.naturalHeight,src:e.currentSrc||e.src})")
                        if dimensions["w"] >= 512 and dimensions["h"] >= 512:
                            # Screenshot the actual rendered image element, avoiding private API calls.
                            blob = img.screenshot(timeout=15000)
                            with Image.open(BytesIO(blob)) as check:
                                check.verify()
                            print(f"[RAPHAEL] captured rendered output {dimensions['w']}x{dimensions['h']}")
                            return blob
                    except Exception as exc:
                        last_error = str(exc)
                        continue
            raise RuntimeError("Timed out waiting for a rendered Raphael image. " + last_error[:300])
        finally:
            context.close()
            browser.close()
