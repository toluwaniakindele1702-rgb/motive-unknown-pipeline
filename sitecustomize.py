"""Runtime compatibility patches for provider-side Groq behavior."""

from __future__ import annotations

try:
    from groq.resources.chat.completions import Completions

    _original_create = Completions.create

    def _patched_create(self, *args, **kwargs):
        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            json_schema = response_format.get("json_schema")
            # SEO is parsed and validated by pipeline.py after the call. Avoid
            # Groq's brittle server-side schema enforcement entirely for this
            # payload; this prevents a malformed extra/empty alternate title
            # from killing a run after all images have already been generated.
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

# Apply the topic-selection hardening inside the runner workspace before pipeline.py is imported.
# This keeps the production workflow self-healing even if the main pipeline file is restored
# from an older commit. The patch is idempotent and only changes the topic-selection rules.
try:
    from pathlib import Path
    import runpy
    _topic_patch = Path("harden_topic_selection.py")
    if _topic_patch.exists():
        runpy.run_path(str(_topic_patch), run_name="__topic_hardening__")
except Exception as exc:
    print(f"[TOPIC HARDENING] Could not apply runtime topic patch: {exc}")

# Keep the burned-in Shorts captions compact. pipeline.py currently renders
# them through ffmpeg's subtitles filter at FontSize=22; intercept only that
# exact Shorts style token so the long-form renderer remains untouched.
try:
    import subprocess

    _original_subprocess_run = subprocess.run

    def _shorts_caption_size_patch(*args, **kwargs):
        patched_args = list(args)
        if patched_args:
            command = patched_args[0]
            if isinstance(command, (list, tuple)):
                patched_args[0] = [
                    str(item).replace("FontSize=22", "FontSize=18").replace("Outline=3", "Outline=2")
                    if isinstance(item, str) else item
                    for item in command
                ]
            elif isinstance(command, str):
                patched_args[0] = command.replace("FontSize=22", "FontSize=18").replace("Outline=3", "Outline=2")
        elif isinstance(kwargs.get("args"), (list, tuple)):
            kwargs = dict(kwargs)
            kwargs["args"] = [
                str(item).replace("FontSize=22", "FontSize=18").replace("Outline=3", "Outline=2")
                if isinstance(item, str) else item
                for item in kwargs["args"]
            ]
        elif isinstance(kwargs.get("args"), str):
            kwargs = dict(kwargs)
            kwargs["args"] = kwargs["args"].replace("FontSize=22", "FontSize=18").replace("Outline=3", "Outline=2")
        return _original_subprocess_run(*patched_args, **kwargs)

    subprocess.run = _shorts_caption_size_patch
    print("[SHORTS CAPTIONS] Compact caption style enabled: FontSize=18, Outline=2.")
except Exception as exc:
    print(f"[SHORTS CAPTIONS] Could not install caption-size patch: {exc}")
