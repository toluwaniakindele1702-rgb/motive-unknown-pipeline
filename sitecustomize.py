"""Runtime compatibility patches for provider-side Groq behavior."""

from __future__ import annotations

try:
    from groq.resources.chat.completions import Completions

    _original_create = Completions.create

    def _patched_create(self, *args, **kwargs):
        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            json_schema = response_format.get("json_schema")
            if isinstance(json_schema, dict) and json_schema.get("name") == "motive_unknown_seo":
                fallback_kwargs = dict(kwargs)
                fallback_kwargs["response_format"] = {"type": "json_object"}
                print("[GROQ COMPAT] SEO schema forced to generic JSON for server compatibility.")
                return _original_create(self, *args, **fallback_kwargs)
            if isinstance(json_schema, dict):
                schema = json_schema.get("schema")
                if isinstance(schema, dict):
                    properties = schema.get("properties")
                    if isinstance(properties, dict):
                        alternate_titles = properties.get("alternate_titles")
                        if isinstance(alternate_titles, dict) and alternate_titles.get("maxItems") == 2:
                            alternate_titles["maxItems"] = 3
            try:
                return _original_create(self, *args, **kwargs)
            except Exception as exc:
                text = str(exc).lower()
                schema_error = (
                    "response_format" in text
                    or "json_schema" in text
                    or "failed to match any of the schemas" in text
                    or "invalid_json" in text
                )
                status = getattr(exc, "status_code", None)
                if schema_error or status == 400:
                    fallback_kwargs = dict(kwargs)
                    fallback_kwargs["response_format"] = {"type": "json_object"}
                    print("[GROQ COMPAT] Structured output rejected; retrying once with generic JSON.")
                    return _original_create(self, *args, **fallback_kwargs)
                raise
        return _original_create(self, *args, **kwargs)

    Completions.create = _patched_create
except Exception as exc:
    print(f"[GROQ COMPAT] Could not install compatibility patch: {exc}")

try:
    from pathlib import Path
    import runpy
    _topic_patch = Path("harden_topic_selection.py")
    if _topic_patch.exists():
        runpy.run_path(str(_topic_patch), run_name="__topic_hardening__")
    _topic_gate = Path("topic_originality_gate.py")
    if _topic_gate.exists():
        runpy.run_path(str(_topic_gate), run_name="__topic_originality_gate__")
        print("[TOPIC GATE] Persisted-history originality gate installed before topic selection.")
except Exception as exc:
    print(f"[TOPIC HARDENING] Could not apply runtime topic patch: {exc}")

try:
    import subprocess
    _original_subprocess_run = subprocess.run

    def _shorts_caption_size_patch(*args, **kwargs):
        patched_args = list(args)
        if patched_args:
            command = patched_args[0]
            if isinstance(command, (list, tuple)):
                patched_args[0] = [str(item).replace("FontSize=22", "FontSize=18").replace("Outline=3", "Outline=2") if isinstance(item, str) else item for item in command]
            elif isinstance(command, str):
                patched_args[0] = command.replace("FontSize=22", "FontSize=18").replace("Outline=3", "Outline=2")
        elif isinstance(kwargs.get("args"), (list, tuple)):
            kwargs = dict(kwargs)
            kwargs["args"] = [str(item).replace("FontSize=22", "FontSize=18").replace("Outline=3", "Outline=2") if isinstance(item, str) else item for item in kwargs["args"]]
        elif isinstance(kwargs.get("args"), str):
            kwargs = dict(kwargs)
            kwargs["args"] = kwargs["args"].replace("FontSize=22", "FontSize=18").replace("Outline=3", "Outline=2")
        return _original_subprocess_run(*patched_args, **kwargs)

    subprocess.run = _shorts_caption_size_patch
    print("[SHORTS CAPTIONS] Compact caption style enabled: FontSize=18, Outline=2.")
except Exception as exc:
    print(f"[SHORTS CAPTIONS] Could not install caption-size patch: {exc}")

try:
    _pipeline_path = Path("pipeline.py")
    if _pipeline_path.exists():
        _source = _pipeline_path.read_text(encoding="utf-8")
        _old = '''        if count_words(text) < 25:\n            raise RuntimeError(f"Retention repair made scene {scene_id} too short.")\n        by_id[scene_id]["narration"] = text\n'''
        _new = '''        if count_words(text) < 25:\n            fallback_text = normalize_spaces(str(original.get("narration", "")))\n            if count_words(fallback_text) >= 18:\n                print(f"[RETENTION] scene {scene_id} rewrite was too short; keeping original narration.")\n                text = fallback_text\n            else:\n                raise RuntimeError(f"Retention repair made scene {scene_id} too short and no valid fallback exists.")\n        by_id[scene_id]["narration"] = text\n'''
        if _old in _source and _new not in _source:
            _pipeline_path.write_text(_source.replace(_old, _new, 1), encoding="utf-8")
            print("[RETENTION PATCH] Short retention rewrites now fall back to the original valid scene.")
except Exception as exc:
    print(f"[RETENTION PATCH] Could not install retention safety patch: {exc}")

# Hard script minimum: repair passes must reach 1700 words, not merely get within tolerance.
try:
    _pipeline_path = Path("pipeline.py")
    if _pipeline_path.exists():
        _source = _pipeline_path.read_text(encoding="utf-8")
        _needle = '''        final_words = _script_word_count(repaired)\n\n        # The repair target is preferred, but a valid script that is only slightly\n'''
        _replacement = '''        final_words = _script_word_count(repaired)\n\n        if final_words < SCRIPT_MIN_WORDS:\n            for _hard_pass in range(1, 5):\n                deficit = SCRIPT_MIN_WORDS - final_words\n                if deficit <= 0:\n                    break\n                ranked = sorted(repaired["scenes"], key=lambda _s: count_words(str(_s.get("narration", ""))))\n                made_progress = False\n                for _scene in ranked[:6]:\n                    before = count_words(str(_scene.get("narration", "")))\n                    add_words = max(8, min(28, deficit + 6))\n                    _prompt = f"""\nExpand ONLY this narration by about {add_words} useful words. Preserve all facts and meaning.\nDo not remove existing useful information. Do not add filler, scenery, dialogue, repetition, or unsupported claims.\nKeep the narration natural and below 120 words. Return JSON only: {{\\\"narration\\\": \\"updated narration\\\"}}.\n\nTOPIC:\n{json.dumps(topic, ensure_ascii=False)}\n\nSCENE:\n{json.dumps({\\\"id\\\": _scene[\\\"id\\\"], \\"narration\\\": _scene[\\\"narration\\\"]}, ensure_ascii=False)}\n""".strip()\n                    try:\n                        _result = groq_json(GROQ_WRITER_MODEL, [{"role": "user", "content": _prompt}], max_completion_tokens=500, temperature=0.25, attempts=2)\n                        _candidate = normalize_spaces(str(_result.get("narration", "")))\n                        if count_words(_candidate) > 120:\n                            _candidate = _fit_narration_to_limit(_candidate, 120)\n                        after = count_words(_candidate)\n                        if 45 <= after <= 120 and after > before:\n                            _scene["narration"] = _candidate\n                            final_words = _script_word_count(repaired)\n                            made_progress = True\n                            atomic_write_json(SCRIPT_PATH, repaired)\n                            _save_current_json(CURRENT_SCRIPT_PATH, repaired)\n                            print(f"[SCRIPT] hard top-up pass {_hard_pass}: scene {_scene['id']} {before}->{after}; total ~{final_words}")\n                            if final_words >= SCRIPT_MIN_WORDS:\n                                break\n                    except Exception as _exc:\n                        print(f"[SCRIPT] hard top-up scene {_scene.get('id')} failed: {_exc}")\n                if final_words >= SCRIPT_MIN_WORDS or not made_progress:\n                    break\n\n        # The repair target is preferred, but a valid script that is only slightly\n'''
        if _needle in _source and "if final_words < SCRIPT_MIN_WORDS:\n            for _hard_pass" not in _source:
            _pipeline_path.write_text(_source.replace(_needle, _replacement, 1), encoding="utf-8")
            print("[SCRIPT PATCH] Hard 1700-word top-up installed.")
except Exception as exc:
    print(f"[SCRIPT PATCH] Could not install hard script-length patch: {exc}")

# SEO fail-safe: a malformed/unsupported Groq structured response must never abort a
# completed production. If the model-side SEO call fails, build valid metadata locally
# from the already-approved topic instead of losing Cloudflare/TTS/render work.
try:
    import pipeline as _relic_pipeline

    _original_build_seo = _relic_pipeline.build_seo

    def _safe_build_seo(topic, script, research):
        try:
            return _original_build_seo(topic, script, research)
        except Exception as exc:
            question = str(topic.get("question") or "Why Does This Happen?").strip()
            title = question[:70].rstrip()
            if len(title) < 18:
                title = "Why Does This Happen?"
            words = [w.lower() for w in __import__("re").findall(r"[a-zA-Z]{4,}", question)]
            tags = []
            for tag in [*words, "everyday life", "psychology", "human behavior", "curiosity", "Relic Loop"]:
                tag = " ".join(str(tag).split())
                if tag and tag.lower() not in {x.lower() for x in tags}:
                    tags.append(tag)
            tags = tags[:15]
            while len(tags) < 12:
                tags.append(["why", "how things work", "interesting facts", "explained"][len(tags) % 4])
            keywords = list(dict.fromkeys(words[:6] + ["everyday life", "human behavior"]))[:8]
            headline = "THE HIDDEN REASON"
            description = (
                f"Have you ever wondered {question.rstrip('?').lower()}? This Relic Loop episode explains "
                "the evidence-backed reason behind this familiar everyday experience, including the "
                "mechanisms, surprising details, and what we can learn from them.\n\n"
                "Sources and research notes were used to build the explanation."
            )
            print(f"[SEO FAILSAFE] Groq SEO failed after production work; using deterministic metadata: {exc}")
            return {
                "title": title,
                "alternate_titles": [title, f"The Real Reason: {title}"[:70]],
                "description": description,
                "tags": tags,
                "primary_keywords": keywords,
                "thumbnail_headline": headline,
            }

    _relic_pipeline.build_seo = _safe_build_seo
    print("[SEO FAILSAFE] Non-fatal local SEO fallback installed.")
except Exception as exc:
    print(f"[SEO FAILSAFE] Could not install SEO fallback: {exc}")
