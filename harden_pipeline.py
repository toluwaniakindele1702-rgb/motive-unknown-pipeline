from pathlib import Path

pipeline = Path('pipeline.py')
text = pipeline.read_text(encoding='utf-8')
anchor = 'CURRENT_SHORTS_PATH = CURRENT_RUN_DIR / "shorts.json"\n'
insert = '''\nPROVIDER_HEALTH_PATH = STATE_DIR / "provider_health.json"\n\ndef _provider_health() -> dict[str, Any]:\n    data = load_json(PROVIDER_HEALTH_PATH, {})\n    return data if isinstance(data, dict) else {}\n\ndef _provider_is_blocked(provider: str) -> bool:\n    entry = _provider_health().get(provider)\n    if not isinstance(entry, dict):\n        return False\n    until = str(entry.get("disabled_until", ""))\n    if not until:\n        return False\n    try:\n        return datetime.now(timezone.utc) < datetime.fromisoformat(until)\n    except ValueError:\n        return False\n\ndef _block_provider(provider: str, reason: str) -> None:\n    from datetime import timedelta\n    until = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)\n    data = _provider_health()\n    data[provider] = {\n        "disabled_until": until.isoformat(),\n        "reason": str(reason)[:1800],\n        "updated_at": datetime.now(timezone.utc).isoformat(),\n    }\n    atomic_write_json(PROVIDER_HEALTH_PATH, data)\n    print(f"[PROVIDER HEALTH] {provider} blocked until {until.isoformat()}")\n\ndef _provider_failure_should_persist(provider: str, message: str) -> bool:\n    lower = str(message).lower()\n    if provider == "cloudflare":\n        return "429" in lower or "daily free allocation" in lower or "10,000 neurons" in lower\n    if provider == "puter":\n        return "insufficient" in lower and "credit" in lower\n    if provider in {"huggingface", "replicate"}:\n        return "not configured" in lower\n    return False\n'''
if 'PROVIDER_HEALTH_PATH = STATE_DIR / "provider_health.json"' not in text:
    if anchor not in text:
        raise SystemExit('provider health anchor missing')
    text = text.replace(anchor, anchor + insert + '\n', 1)

old = '''    for provider in IMAGE_PROVIDER_ORDER:\n        if provider in disabled:\n            continue\n        attempted.append(provider)'''
new = '''    for provider in IMAGE_PROVIDER_ORDER:\n        if provider in disabled:\n            continue\n        if _provider_is_blocked(provider):\n            print(f"[IMAGE FALLBACK] skipping {provider}: provider health cooldown")\n            disabled.add(provider)\n            continue\n        attempted.append(provider)'''
if old in text:
    text = text.replace(old, new, 1)

old = '''        except ImageProviderError as exc:\n            print(f"[IMAGE FALLBACK] {provider} failed: {exc}")\n            if exc.disable_for_run:\n                disabled.add(provider)\n                setattr(_generate_image_with_fallback, "_disabled", disabled)\n                print(f"[IMAGE FALLBACK] disabling {provider} for the rest of this run.")'''
new = '''        except ImageProviderError as exc:\n            print(f"[IMAGE FALLBACK] {provider} failed: {exc}")\n            if _provider_failure_should_persist(provider, str(exc)):\n                _block_provider(provider, str(exc))\n            if exc.disable_for_run:\n                disabled.add(provider)\n                setattr(_generate_image_with_fallback, "_disabled", disabled)\n                print(f"[IMAGE FALLBACK] disabling {provider} for the rest of this run.")'''
if old in text:
    text = text.replace(old, new, 1)

pipeline.write_text(text, encoding='utf-8')

worker = Path('local_image_worker.py')
w = worker.read_text(encoding='utf-8')
old = '''    result = PIPELINE(\n        compact,\n        num_inference_steps=steps,\n        guidance_scale=8.0,\n        lcm_origin_steps=50,\n        width=width,\n        height=height,\n        generator=generator,\n    )'''
new = '''    negative_prompt = (\n        "text, letters, words, captions, subtitles, logos, watermark, fake writing, "\n        "random signage, UI screenshot, poster, collage, split screen, duplicate objects, "\n        "extra limbs, malformed hands, distorted face, deformed body, inappropriate body-focused framing, "\n        "unrelated person, unrelated animal, gore, blood, injury, violence, horror, "\n        "photorealistic stock photo, vintage textbook art, sepia, muddy composition, "\n        "blurry, low detail, clutter, decorative filler"\n    )\n    result = PIPELINE(\n        compact,\n        negative_prompt=negative_prompt,\n        num_inference_steps=steps,\n        guidance_scale=8.0,\n        lcm_origin_steps=50,\n        width=width,\n        height=height,\n        generator=generator,\n    )'''
if 'negative_prompt = (' not in w:
    if old not in w:
        raise SystemExit('local worker generation call missing')
    w = w.replace(old, new, 1)
worker.write_text(w, encoding='utf-8')

workflow = Path('.github/workflows/daily_video.yml')
y = workflow.read_text(encoding='utf-8')
y = y.replace("IMAGE_PROVIDERS: 'cloudflare,puter,huggingface,replicate'", "IMAGE_PROVIDERS: 'cloudflare,puter,huggingface,replicate,local'", 1)
y = y.replace('run_work/output/short_*.srt\n            state/current_run/shorts.json', 'run_work/output/short_*.srt\n            run_work/output/*.mp4\n            state/current_run/shorts.json\n            state/provider_health.json', 1)
workflow.write_text(y, encoding='utf-8')
print('Hardening patch applied.')
