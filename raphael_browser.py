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

RAPHAEL_URL = os.getenv("RAPHAEL_URL", "https://raphael.app/gpt-image-2-6-sunburst")
MODEL_PREFERENCE = [
    x.strip() for x in os.getenv(
        "RAPHAEL_MODEL_PREFERENCE",
        "GPT Image 2.6 Sunburst,GPT Image 2.6 Flare,GPT Image 2.5 Flare,GPT Image 2,Seedream 5.0 Pro,Seedream 3.5 Pro"
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


def _ratio_ui_diagnostics(page) -> str:
    """Capture visible controls to diagnose UI changes without bypassing verification."""
    rows=[]
    controls=page.locator("button,[role='button'],[role='combobox'],[role='radio'],[role='option']")
    for i in range(min(controls.count(),100)):
        el=controls.nth(i)
        try:
            if not el.is_visible(): continue
            rows.append(el.evaluate("""e => ({tag:e.tagName,text:(e.innerText||e.textContent||'').trim().slice(0,90),aria:e.getAttribute('aria-label'),title:e.getAttribute('title'),role:e.getAttribute('role'),expanded:e.getAttribute('aria-expanded'),html:e.outerHTML.slice(0,300)})"""))
        except Exception: pass
    return json.dumps(rows[:45],ensure_ascii=False)


def _select_portrait_ratio(page) -> bool:
    """Discover the current ratio control and positively select 9:16."""
    controls=page.locator("button,[role='button'],[role='combobox']")
    ranked=[]
    for i in range(min(controls.count(),120)):
        el=controls.nth(i)
        try:
            if not el.is_visible() or not el.is_enabled(): continue
            meta=el.evaluate("""e => ({text:(e.innerText||e.textContent||'').trim(),aria:e.getAttribute('aria-label')||'',title:e.getAttribute('title')||'',html:e.outerHTML.slice(0,450)})""")
            blob=" ".join([meta["text"],meta["aria"],meta["title"],meta["html"]]).lower()
            score=(8 if "aspect" in blob or "ratio" in blob else 0)+(5 if "1:1" in blob or "square" in blob or "9:16" in blob else 0)+(4 if "portrait" in blob or "dimension" in blob else 0)
            if score: ranked.append((score,i))
        except Exception: pass
    for _,idx in sorted(ranked,reverse=True):
        try:
            controls.nth(idx).click(timeout=3000)
            page.wait_for_timeout(400)
            for selector in ["[role='option']","[role='radio']","[role='menuitem']","button"]:
                opts=page.locator(selector).filter(has_text=re.compile(r"^\\s*(?:Portrait\\s*)?9\\s*:\\s*16\\s*$",re.I))
                for j in range(min(opts.count(),10)):
                    opt=opts.nth(j)
                    try:
                        if opt.is_visible() and opt.is_enabled():
                            opt.click(timeout=3000)
                            page.wait_for_timeout(500)
                            print("[RAPHAEL] selected 9:16 via discovered ratio control")
                            return True
                    except Exception: pass
            portrait=page.get_by_text(re.compile(r"^\\s*Portrait\\s*$",re.I))
            for j in range(min(portrait.count(),5)):
                try:
                    if portrait.nth(j).is_visible():
                        portrait.nth(j).click(timeout=3000)
                        page.wait_for_timeout(500)
                        if page.get_by_text(re.compile(r"9\\s*:\\s*16")).count():
                            print("[RAPHAEL] selected Portrait option")
                            return True
                except Exception: pass
        except Exception: pass
    print("[RAPHAEL] ratio selector unresolved; live UI controls="+_ratio_ui_diagnostics(page))
    return False


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
            if not _select_portrait_ratio(page):
                # Source dimensions are normalized by the image router after capture.
                # Do not reject a valid Raphael result solely because ratio UI is absent.
                print("[RAPHAEL] portrait selector unavailable; continuing with native output dimensions.")

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

            # Remember pre-existing reference/gallery images so they cannot be
            # mistaken for the newly generated result.
            before = set()
            imgs = page.locator("img")
            for i in range(imgs.count()):
                try:
                    before.add(imgs.nth(i).evaluate("(e) => e.currentSrc || e.src || ''"))
                except Exception:
                    pass

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
                candidates = []
                for idx in range(imgs.count()):
                    img = imgs.nth(idx)
                    try:
                        if not img.is_visible():
                            continue
                        meta = img.evaluate("""e => ({
                          w:e.naturalWidth,h:e.naturalHeight,src:e.currentSrc||e.src||'',
                          complete:e.complete,area:(e.getBoundingClientRect().width*e.getBoundingClientRect().height)
                        })""")
                        if meta["w"] >= 1 and meta["h"] >= 1 and meta["src"] not in before:
                            candidates.append((meta["area"], idx, meta))
                    except Exception as exc:
                        last_error = str(exc)
                for _, idx, meta in sorted(candidates, reverse=True):
                    try:
                        img = imgs.nth(idx)
                        encoded = img.evaluate("""async e => {
                          const src=e.currentSrc||e.src;
                          const response=await fetch(src,{credentials:'include'});
                          if(!response.ok) throw new Error('image fetch HTTP '+response.status);
                          const blob=await response.blob();
                          const bytes=new Uint8Array(await blob.arrayBuffer());
                          let binary='';
                          for(let i=0;i<bytes.length;i+=0x8000)
                            binary+=String.fromCharCode(...bytes.subarray(i,i+0x8000));
                          return {mime:blob.type||'image/png',data:btoa(binary)};
                        }""")
                        import base64
                        blob = base64.b64decode(encoded["data"], validate=True)
                        with Image.open(BytesIO(blob)) as check:
                            check.verify()
                        with Image.open(BytesIO(blob)) as check:
                            width, height = check.size
                        print(f"[RAPHAEL] captured source image {width}x{height}; UI element {meta['w']}x{meta['h']}")
                        if width < 1 or height < 1:
                            raise RuntimeError(f"Raphael output has invalid dimensions ({width}x{height}).")
                        return blob
                    except Exception as exc:
                        last_error = str(exc)
                        if "not portrait" in last_error.lower():
                            # A fresh square result proves the ratio selection did
                            # not apply. Stop immediately and use Cloudflare fallback.
                            raise RuntimeError(last_error) from exc
            raise RuntimeError("Timed out waiting for a new, directly captured Raphael output. " + last_error[:300])
        finally:
            context.close()
            browser.close()
