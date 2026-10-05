"""Runtime compatibility patches for provider-side Groq behavior."""

from __future__ import annotations

try:
    from groq.resources.chat.completions import Completions

    _original_create = Completions.create

    def _patched_create(self, *args, **kwargs):
        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            json_schema = response_format.get("json_schema")
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
                # Groq can reject an otherwise valid SEO request at its strict
                # structured-output boundary. Retry once as generic JSON so the
                # pipeline can parse/validate the returned object itself instead
                # of losing an expensive image/video run.
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
                    print("[GROQ COMPAT] Structured SEO output rejected; retrying once with generic JSON.")
                    return _original_create(self, *args, **fallback_kwargs)
                raise

        return _original_create(self, *args, **kwargs)

    Completions.create = _patched_create
except Exception as exc:
    print(f"[GROQ COMPAT] Could not install compatibility patch: {exc}")
