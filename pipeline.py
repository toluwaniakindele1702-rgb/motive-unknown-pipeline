"""
Relic Loop v6 — curiosity-first automated explainer video factory.

Design goals
------------
- Daily, unattended GitHub Actions execution.
- Curiosity-driven questions across everyday life, animals, science, technology, culture, society, and history.
- GPT-OSS 120B for research/storytelling; GPT-OSS 20B for lightweight
  structuring/SEO tasks.
- Local/open-weight Kokoro TTS (no paid voice API).
- Modern animated-documentary illustrations using a multi-provider fallback chain.
- Visual shots are chosen to explain mechanisms, comparisons, processes, evidence, maps, reactions, and consequences.
- Each narration scene is split into compact visual beats so the picture changes frequently.
- Visual prompts prioritize the exact narrated fact, action, evidence, place, person, or object.
- A persistent style-reference image plus prior-frame references improve visual continuity.
- Motion-comic camera drift, reveal emphasis cards, and subtle original sound design add movement without relying on subtitles.
- No burned-in subtitles.
- Dedicated curiosity-thumbnail generation paired with title packaging and sparse reveal graphics.
- Idempotent stage files so a rerun can skip already-completed stages.
- YouTube uploads default to public for unattended channel publishing.

Required GitHub Secrets
-----------------------
GROQ_API_KEY
YOUTUBE_TOKEN_JSON
YOUTUBE_CLIENT_SECRET_JSON
CLOUDFLARE_ACCOUNT_ID
CLOUDFLARE_API_TOKEN

Optional GitHub Secrets
-----------------------
HUGGINGFACE_TOKEN
REPLICATE_API_TOKEN

Optional GitHub Variables / Secrets
-----------------------------------
YOUTUBE_PRIVACY_STATUS     default: public
KOKORO_VOICE               default: af_bella
KOKORO_SPEED               default: 1.05
CHANNEL_NAME               optional, used in prompts/description

Optional repo assets
--------------------
assets/music/background.mp3  royalty-free / licensed music only
assets/visual_style_reference.jpg.b64  embedded JPEG style reference used as an image-model style anchor

The workflow file supplied with this package runs daily and also supports manual
"voice_test", "visual_test", and "local_image_test" modes.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
import soundfile as sf
from groq import Groq
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
CURRENT_RUN_DIR = STATE_DIR / "current_run"
WORK_DIR = ROOT / "run_work"
SCENE_DIR = WORK_DIR / "scenes"
AUDIO_DIR = WORK_DIR / "audio"
THUMB_DIR = WORK_DIR / "thumbnails"
OUTPUT_DIR = WORK_DIR / "output"

for d in (STATE_DIR, CURRENT_RUN_DIR, WORK_DIR, SCENE_DIR, AUDIO_DIR, THUMB_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_RESEARCH_MODEL = os.environ.get("GROQ_RESEARCH_MODEL", "openai/gpt-oss-120b")
GROQ_WRITER_MODEL = os.environ.get("GROQ_WRITER_MODEL", "openai/gpt-oss-120b")
GROQ_LIGHT_MODEL = os.environ.get("GROQ_LIGHT_MODEL", "openai/gpt-oss-20b")

KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_bella").strip()
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", "1.05"))
BACKGROUND_MUSIC_VOLUME = float(os.environ.get("BACKGROUND_MUSIC_VOLUME", "0.055"))
ENABLE_SOUND_DESIGN = os.environ.get("ENABLE_SOUND_DESIGN", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_EMPHASIS_CARDS = os.environ.get("ENABLE_EMPHASIS_CARDS", "1").strip().lower() not in {"0", "false", "no"}
MOTION_INTENSITY = os.environ.get("MOTION_INTENSITY", "1.0").strip()
PREMIUM_EMPHASIS_PHRASES = ["WHY?", "BUT WHY?", "THE TWIST", "THE REAL REASON", "SO THAT'S WHY"]

# Polished AI illustration generation.
IMAGE_PROVIDER_ORDER = [
    x.strip().lower()
    for x in os.environ.get("IMAGE_PROVIDERS", "cloudflare,huggingface,replicate,local").split(",")
    if x.strip()
]
if not IMAGE_PROVIDER_ORDER:
    IMAGE_PROVIDER_ORDER = ["cloudflare", "huggingface", "replicate", "local"]
IMAGE_PROVIDER = IMAGE_PROVIDER_ORDER[0]
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_IMAGE_MODEL = os.environ.get(
    "CLOUDFLARE_IMAGE_MODEL",
    "@cf/black-forest-labs/flux-2-klein-4b",
).strip()
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
HUGGINGFACE_IMAGE_MODEL = os.environ.get("HUGGINGFACE_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell").strip()
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "").strip()
REPLICATE_IMAGE_MODEL = os.environ.get("REPLICATE_IMAGE_MODEL", "black-forest-labs/flux-1.1-pro").strip()
IMAGE_W = 1024
IMAGE_H = 576
MAX_VISUAL_BEATS_PER_VIDEO = int(os.environ.get("MAX_VISUAL_BEATS_PER_VIDEO", "110"))
VISUAL_BEAT_MIN_DURATION = float(os.environ.get("VISUAL_BEAT_MIN_DURATION", "0.9"))
LOCAL_IMAGE_MODEL = os.environ.get("LOCAL_IMAGE_MODEL", "OpenVINO/LCM_Dreamshaper_v7-int8-ov").strip()
LOCAL_IMAGE_STEPS = int(os.environ.get("LOCAL_IMAGE_STEPS", "4"))
LOCAL_IMAGE_WIDTH = int(os.environ.get("LOCAL_IMAGE_WIDTH", "768"))
LOCAL_IMAGE_HEIGHT = int(os.environ.get("LOCAL_IMAGE_HEIGHT", "512"))
LOCAL_IMAGE_ENV_DIR = WORK_DIR / ".local_image_env"
LOCAL_IMAGE_WORKER = ROOT / "local_image_worker.py"
LOCAL_IMAGE_DEPS = ROOT / "requirements-local-image.txt"
STYLE_REFERENCE_B64 = ROOT / "assets" / "visual_style_reference.jpg.b64"

CHANNEL_NAME = os.environ.get("CHANNEL_NAME", "Relic Loop").strip()
if not CHANNEL_NAME or CHANNEL_NAME.lower() == "motive unknown":
    CHANNEL_NAME = "Relic Loop"
YOUTUBE_PRIVACY_STATUS = os.environ.get("YOUTUBE_PRIVACY_STATUS", "public").strip().lower()
if YOUTUBE_PRIVACY_STATUS not in {"private", "public", "unlisted"}:
    YOUTUBE_PRIVACY_STATUS = "public"

VIDEO_W, VIDEO_H = 1280, 720
VIDEO_FPS = 30
AUDIO_SR = 24000
MUSIC_PATH = ROOT / "assets" / "music" / "background.mp3"

SCENE_MIN = 18
SCENE_MAX = 38
SCRIPT_MIN_WORDS = 1700
SCRIPT_MAX_WORDS = 2600

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Small, reusable color palette for a child-friendly illustrated look.
PAPER = (248, 244, 232)
INK = (38, 37, 35)
WHITE = (255, 255, 255)
BLACK = (15, 15, 15)
SKY = (220, 236, 246)
GRASS = (175, 203, 146)
SAND = (228, 199, 143)
STONE = (166, 163, 150)
RED = (185, 75, 63)
BLUE = (75, 113, 158)
GOLD = (214, 167, 66)
BROWN = (128, 91, 57)
GREEN = (77, 126, 81)
PURPLE = (117, 88, 138)

# ---------------------------------------------------------------------------
# Files / persistence
# ---------------------------------------------------------------------------
HISTORY_PATH = STATE_DIR / "content_history.json"
RUN_META_PATH = STATE_DIR / "last_run.json"
RESEARCH_PATH = WORK_DIR / "research.txt"
TOPIC_PATH = WORK_DIR / "topic.json"
STORY_PATH = WORK_DIR / "story.json"
SEO_PATH = WORK_DIR / "seo.json"
SCRIPT_PATH = WORK_DIR / "script.json"
MANIFEST_PATH = WORK_DIR / "manifest.json"


# Small text-only artifacts survive a fresh GitHub Actions runner.
CURRENT_TOPIC_PATH = CURRENT_RUN_DIR / "topic.json"
CURRENT_RESEARCH_PATH = CURRENT_RUN_DIR / "research.txt"
CURRENT_STORY_PATH = CURRENT_RUN_DIR / "story.json"
CURRENT_SCRIPT_PATH = CURRENT_RUN_DIR / "script.json"
CURRENT_SEO_PATH = CURRENT_RUN_DIR / "seo.json"
CURRENT_UPLOAD_PATH = CURRENT_RUN_DIR / "upload.json"


def atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_history() -> dict[str, Any]:
    data = load_json(HISTORY_PATH, {"videos": []})
    if not isinstance(data, dict) or not isinstance(data.get("videos"), list):
        return {"videos": []}
    return data


def save_history(data: dict[str, Any]) -> None:
    atomic_write_json(HISTORY_PATH, data)


def checkpoint(stage: str, **extra: Any) -> None:
    data = {
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    atomic_write_json(RUN_META_PATH, data)
    print(f"[CHECKPOINT] {stage}")


def _save_current_json(path: Path, data: Any) -> None:
    atomic_write_json(path, data)


def _save_current_text(path: Path, content: str) -> None:
    atomic_write_text(path, content)


def _load_current_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return load_json(path, None)


def _load_current_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def clear_current_run() -> None:
    for path in CURRENT_RUN_DIR.glob("*"):
        if path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                print(f"[STATE] Could not remove {path}: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def require_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment secret: {name}")
    return value


def clean_json_text(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def extract_last_json_object(raw: str) -> dict[str, Any]:
    """Find the last balanced JSON object that parses successfully."""
    candidates: list[str] = []
    stack = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}" and stack:
            stack -= 1
            if stack == 0 and start is not None:
                candidates.append(raw[start : i + 1])
                start = None
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    raise RuntimeError("No valid JSON object found in model response.")


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text))


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_wrap_for_image(text: str, width_chars: int) -> list[str]:
    return textwrap.wrap(normalize_spaces(text), width=width_chars, break_long_words=False)


def safe_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    return text[:max_len].strip("_") or "untitled"


def parse_wait_seconds(resp: requests.Response | Exception, default: float = 8.0) -> float:
    if isinstance(resp, requests.Response):
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return max(float(ra) + 1.0, default)
            except ValueError:
                pass
        match = re.search(r"try again in ([\d.]+)s", resp.text or "", flags=re.IGNORECASE)
        if match:
            return max(float(match.group(1)) + 1.0, default)
    return default


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
if not GROQ_API_KEY:
    client: Groq | None = None
else:
    client = Groq(api_key=GROQ_API_KEY)


def _exception_status(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status from Groq SDK exceptions."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    text = str(exc)
    match = re.search(r"\b(400|401|403|404|409|429|500|502|503|504)\b", text)
    return int(match.group(1)) if match else None


def _is_non_retryable(exc: Exception) -> bool:
    status = _exception_status(exc)
    if status in {400, 401, 403, 404}:
        return True
    text = str(exc).lower()
    return any(token in text for token in (
        "does not support", "unsupported parameter", "invalid_request_error",
        "invalid api key", "authentication", "permission denied",
    ))

def _is_rate_limited(exc: Exception) -> bool:
    """Detect Groq rate limits, including daily token (TPD) exhaustion."""
    status = _exception_status(exc)
    text = str(exc).lower()
    return status == 429 or "rate limit" in text or "rate_limit_exceeded" in text or "tokens per day" in text


def _fallback_model(model: str) -> str | None:
    """Use the lighter model when a larger model is temporarily rate-limited."""
    if model == GROQ_LIGHT_MODEL:
        return None
    return GROQ_LIGHT_MODEL.strip() or None


def _retry_wait(exc: Exception, attempt: int, base: float = 8.0) -> float:
    message = str(exc)
    match = re.search(r"try again in ([\d.]+)s", message, flags=re.IGNORECASE)
    if match:
        return max(float(match.group(1)) + 1.0, base * attempt)
    retry_after = getattr(exc, "headers", None)
    if retry_after and hasattr(retry_after, "get"):
        value = retry_after.get("retry-after")
        if value:
            try:
                return max(float(value) + 1.0, base * attempt)
            except (TypeError, ValueError):
                pass
    return base * attempt


def _tool_results_text(response: Any) -> str:
    """Extract server-side browser-search results when message.content is empty."""
    try:
        message = response.choices[0].message
    except Exception:
        return ""

    executed = getattr(message, "executed_tools", None)
    if not executed:
        return ""

    chunks: list[str] = []
    for tool in executed:
        try:
            if hasattr(tool, "model_dump"):
                data = tool.model_dump()
            elif isinstance(tool, dict):
                data = tool
            else:
                data = vars(tool)
        except Exception:
            data = {"tool": str(tool)}

        results = data.get("search_results") if isinstance(data, dict) else None
        if isinstance(results, dict):
            results = results.get("results", [])
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                url = str(item.get("url", "")).strip()
                content = str(item.get("content", "")).strip()
                if title or content:
                    chunks.append(f"TITLE: {title}\nURL: {url}\nSNIPPET: {content}")

    return "\n\n".join(chunks).strip()


def groq_call(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_completion_tokens: int,
    temperature: float = 0.6,
    attempts: int = 5,
) -> str:
    """Normal Groq call with intelligent retry behavior."""
    if client is None:
        raise RuntimeError("GROQ_API_KEY is required for this step.")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            # Groq's JSON response mode requires the prompt messages to explicitly
            # mention JSON. Add that guard here so every structured call is safe,
            # including future prompts that forget the literal word.
            safe_messages = [{"role": "system", "content": "Return valid JSON."}, *messages]
            response = client.chat.completions.create(
                model=model,
                messages=safe_messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort="low",
            )
            content = response.choices[0].message.content or ""
            if content.strip():
                return content.strip()
            raise RuntimeError(f"Groq returned an empty response for model {model}.")
        except Exception as exc:
            last_error = exc
            if _is_non_retryable(exc):
                raise RuntimeError(f"Groq rejected the request for {model}: {exc}") from exc
            fallback = _fallback_model(model) if _is_rate_limited(exc) else None
            if fallback:
                print(f"[GROQ FALLBACK] {model} is rate-limited; switching to {fallback} instead of burning the run.")
                return groq_call(
                    fallback, messages,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    attempts=2,
                )
            if attempt >= attempts:
                break
            wait = _retry_wait(exc, attempt)
            print(f"[GROQ RETRY] {model} attempt {attempt}/{attempts}: {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Groq call failed after {attempts} attempts: {last_error}")


def groq_browser_search(
    model: str,
    prompt: str,
    *,
    max_completion_tokens: int = 2800,
    attempts: int = 3,
) -> str:
    """Run browser search WITHOUT structured output/citation_options.

    Groq documents that Browser Search is not compatible with structured outputs.
    We therefore keep browsing as a plain-text stage and structure the result in a
    separate Groq call. If the SDK returns empty message.content but exposes the
    executed search results, those results are used as the search payload.
    """
    if client is None:
        raise RuntimeError("GROQ_API_KEY is required for this step.")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_completion_tokens,
                temperature=0.4,
                reasoning_effort="low",
                tool_choice="required",
                tools=[{"type": "browser_search"}],
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content

            tool_text = _tool_results_text(response)
            if tool_text:
                print("[GROQ SEARCH] message.content was empty; using executed browser-search results.")
                return tool_text

            raise RuntimeError("Groq browser-search call returned neither final text nor search-result payload.")
        except Exception as exc:
            last_error = exc
            if _is_non_retryable(exc):
                raise RuntimeError(f"Groq browser-search request was rejected: {exc}") from exc
            fallback = _fallback_model(model) if _is_rate_limited(exc) else None
            if fallback:
                print(f"[GROQ SEARCH FALLBACK] {model} is rate-limited; switching to {fallback}.")
                return groq_browser_search(
                    fallback, prompt,
                    max_completion_tokens=max_completion_tokens,
                    attempts=2,
                )
            if attempt >= attempts:
                break
            wait = _retry_wait(exc, attempt, base=10.0)
            print(f"[GROQ SEARCH RETRY] {model} attempt {attempt}/{attempts}: {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Groq browser-search failed after {attempts} attempts: {last_error}")


def groq_json(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_completion_tokens: int,
    temperature: float,
    attempts: int = 3,
) -> dict[str, Any]:
    """Structured JSON call. Never combine this with Browser Search."""
    if client is None:
        raise RuntimeError("GROQ_API_KEY is required for this step.")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            safe_messages = [{"role": "system", "content": "Return valid JSON."}, *messages]
            response = client.chat.completions.create(
                model=model,
                messages=safe_messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise RuntimeError("Groq JSON call returned an empty response.")
            return extract_last_json_object(clean_json_text(raw))
        except Exception as exc:
            last_error = exc
            if _is_non_retryable(exc):
                raise RuntimeError(f"Groq JSON request was rejected for {model}: {exc}") from exc
            fallback = _fallback_model(model) if _is_rate_limited(exc) else None
            if fallback:
                print(f"[GROQ JSON FALLBACK] {model} is rate-limited; switching to {fallback}.")
                return groq_json(
                    fallback, messages,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    attempts=2,
                )
            if attempt >= attempts:
                break
            wait = _retry_wait(exc, attempt)
            print(f"[GROQ JSON RETRY] {model} attempt {attempt}/{attempts}: {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Groq JSON call failed after {attempts} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Topic scouting / research
# ---------------------------------------------------------------------------
TOPIC_PROMPT = """
You are the topic producer for Relic Loop, a curiosity-first YouTube channel built around
things people actually notice, do, use, hear about, or wonder about.

CONTENT MIX:
Relic Loop must NOT become a plant/animal/science-only channel and must NOT become a history-only
channel. Keep the center of gravity on everyday life and familiar human experiences. As a rough
creative mix across many uploads, favor:
- everyday objects, routines, places, habits, designs, rules, foods, transport, technology and
  ordinary situations: about 55-65%
- human behavior, psychology, culture and society: about 15-20%
- relatable history and famous historical questions: about 10-15%
- animals/nature and science: about 10-15% combined

These are guidance for variety, not rigid quotas.

The best Relic Loop topic starts with something the viewer already recognizes and then exposes a
hidden reason, surprising consequence, strange design choice, forgotten origin, social behavior,
unexpected chain of events, or counterintuitive explanation.

Think like a great curiosity channel: "You see this all the time. But why is it like that?"
The topic should make the viewer want the answer BEFORE they know the answer.

HIGH-VALUE TOPIC SHAPES:
- ordinary thing + strange design choice
- common habit + hidden reason
- familiar situation + unexpected chain of events
- everyday object + surprising origin
- common technology + overlooked reason it works that way
- social behavior + "why do people do this?"
- food/place/transport/custom + strange explanation
- famous historical event/person + one relatable unanswered question
- animal behavior + a genuinely surprising consequence
- science only when it explains something people actually notice in everyday life

Examples of the KIND of question to seek:
- Why do supermarkets put the things you need at the back?
- Why do elevators have mirrors?
- Why do we say "bless you" after someone sneezes?
- Why are keyboards arranged in that weird order?
- Why do traffic lights use red, yellow, and green?
- Why does popcorn suddenly explode?
- Why do we sometimes feel like our phone vibrated when it didn't?
- Why do some countries drive on the left?
- Why do hotel rooms skip certain floor numbers?
- Why did people start putting pockets in clothes?
- Why do airplanes dim the cabin lights before landing?
- Why do people suddenly copy each other's accents?
- Why did a familiar everyday rule or custom become normal?
- Why did a famous historical decision make sense at the time?

Do NOT simply turn these examples into future videos. Generate fresh questions in the same
curiosity territory.

AVOID:
- obscure trivia that needs a long history lesson before it becomes interesting
- generic biographies
- broad "the history of X" topics
- textbook science lectures
- topics whose only hook is "here are 10 facts"
- fake mysteries, conspiracies, paranormal claims presented as fact
- medical diagnosis/advice
- body-comparison or appearance-ideal content
- repetitive plant/animal episodes
- topics with no concrete visual story

A strong topic should support a real story: hook -> mystery -> first answer -> complication ->
new question -> deeper explanation -> surprising consequence -> satisfying payoff.

Return exactly:
{
  "question": "one specific curiosity question",
  "topic": "short topic label",
  "category": "everyday / human behavior / technology / culture & society / history / animals / science",
  "era": "time/setting label, or modern day",
  "why_curious": "2-4 sentences explaining why an ordinary viewer would care",
  "curiosity_gap": "one sentence describing the obvious assumption versus the hidden explanation",
  "curiosity_score": 8,
  "search_angles": ["angle 1", "angle 2", "angle 3", "angle 4"]
}
""".strip()

def choose_topic(history: dict[str, Any]) -> dict[str, Any]:
    previous = []
    for item in history.get("videos", [])[-60:]:
        q = item.get("question") or item.get("topic") or item.get("title")
        if q:
            previous.append(q)
    history_text = "\n".join(f"- {x}" for x in previous) or "(no previous videos recorded)"

    search_prompt = f"""
Search the web for fresh curiosity-first YouTube topic ideas for Relic Loop.

Prioritize everyday life, familiar objects, routines, places, habits, technology, food,
transport, social behavior, customs, and things people encounter without thinking about them.
Also collect some relatable history, animals/nature, and science ideas for variety.

Use the kinds of curiosity patterns common to large explainer channels:
a familiar thing -> a weird detail -> a "why/how?" -> an evidence-backed explanation -> a
surprising consequence or deeper reveal.

Do NOT return a list of generic science questions. We want topics that feel like things happening
around the viewer's life. Avoid conspiracies, paranormal claims, medical diagnosis/advice,
appearance-ideal content, generic biographies, listicles, and broad topics with no natural question.

Do not repeat these recent channel questions:
{history_text}

Return search findings with the subject, suggested question, evidence, and useful visual angles.
""".strip()

    search_findings = groq_browser_search(
        GROQ_LIGHT_MODEL,
        search_prompt,
        max_completion_tokens=2800,
        attempts=3,
    )

    selection_base = f"""
You are the final topic selector for Relic Loop, a curiosity-first explainer channel.

Choose ONE topic from the findings below.

Everyday life and familiar human experiences should be the default center of gravity.
History, animals, and science are supporting categories, not the channel's identity.

Select something familiar enough to recognize in the first few seconds, but with a non-obvious
reason, consequence, origin, design choice, behavior, or chain of events.

Strong patterns:
- ordinary object/place/routine + hidden reason
- everyday behavior + surprising explanation
- common technology/design + overlooked reason
- familiar food/transport/custom + unexpected origin or function
- ordinary situation + surprising chain of events
- human behavior + contradiction between what people assume and what happens
- famous historical subject + relatable question
- animal/nature/science + a real-world curiosity people can picture

Reject broad subjects, generic biographies, simple event summaries, fake mysteries,
conspiracies/paranormal claims presented as fact, medical diagnosis/advice,
body-comparison/appearance-ideal framing, and claims the sources cannot support.

Quality checks:
0. The topic can support a strong title + thumbnail pairing and at least one visually obvious mystery.
0b. The explanation can be demonstrated visually, not just described verbally.
1. Familiar subject.
2. Immediate curiosity gap.
3. Satisfying evidence-backed answer.
4. Strong visual explanation potential.
5. At least 8 useful reveals/steps without filler.

Do not repeat or closely imitate previous questions:
{history_text}

WEB FINDINGS:
{search_findings}

Return exactly:
{{
  "question": "one specific curiosity-first question",
  "topic": "short topic label",
  "category": "everyday / animals / science / human mind / technology / culture & society / history",
  "era": "time/setting label, or modern day",
  "why_curious": "2-4 sentences explaining the curiosity",
  "curiosity_gap": "one sentence describing the viewer's assumption versus the hidden explanation",
  "curiosity_score": 8,
  "search_angles": ["angle 1", "angle 2", "angle 3", "angle 4"]
}}
""".strip()

    last_data = None
    for attempt in range(1, 4):
        extra = (
            "\n\nRETRY: reject anything generic. Pick a more familiar subject with a sharper "
            "why/how question, stronger explanation, and better visual payoff."
            if attempt > 1 else ""
        )
        data = groq_json(
            GROQ_LIGHT_MODEL,
            [{"role": "user", "content": selection_base + extra}],
            max_completion_tokens=1400,
            temperature=0.72,
            attempts=2,
        )
        last_data = data
        required = ("question","topic","category","era","why_curious","curiosity_gap","search_angles")
        try:
            score = int(data.get("curiosity_score",0))
        except (TypeError,ValueError):
            score = 0
        if all(data.get(k) for k in required) and isinstance(data.get("search_angles"), list) and score >= 8:
            return data
        print(f"[TOPIC] rejected weak candidate on selector attempt {attempt}.")
    raise RuntimeError(
        "Topic selector could not produce a sufficiently curiosity-driven topic after 3 attempts. "
        f"Last keys: {sorted(last_data.keys()) if isinstance(last_data, dict) else []}"
    )


def research_topic(topic: dict[str, Any]) -> str:
    search_prompt = f"""
Research this curiosity question deeply using browser search:
{topic['question']}

Category: {topic.get('category', 'general curiosity')}
Setting/era: {topic.get('era', 'modern day')}
Search angles: {json.dumps(topic.get('search_angles', []), ensure_ascii=False)}

Use reliable sources appropriate to the topic: universities, museums, government or
national institutions, scientific organizations, reputable reference works, strong
reporting, primary sources, and recognized subject-matter institutions.

Find the clearest evidence-backed answer, the mechanism or cause, surprising but
well-supported facts, useful examples/comparisons, common misconceptions, and genuine
uncertainty. Also note things that can be shown visually: processes, anatomy/cutaways,
objects, maps, before/after states, experiments, evidence, scale, or reactions.

Return detailed search findings with source titles and URLs when available. Clearly
distinguish established facts from disputed, uncertain, legendary, or preliminary claims.
""".strip()

    search_findings = groq_browser_search(
        GROQ_RESEARCH_MODEL,
        search_prompt,
        max_completion_tokens=4600,
        attempts=4,
    )

    synthesis_prompt = f"""
You are the lead researcher for Relic Loop.

Turn the browser-search findings into a compact, accurate research dossier for another writer.
Do not invent facts or sources. Preserve uncertainty and disagreement.

CENTRAL QUESTION:
{topic['question']}

CATEGORY:
{topic.get('category', 'general curiosity')}

BROWSER-SEARCH FINDINGS:
{search_findings}

Return plain text with exactly these headings:
1. CORE ANSWER
2. STORY BEATS (10-16 numbered beats)
3. MECHANISM / CAUSE AND EFFECT
4. IMPORTANT PEOPLE / PLACES / OBJECTS / EXAMPLES
5. MISCONCEPTIONS OR CONTRADICTIONS
6. DISPUTES OR UNCERTAINTY
7. VERIFIED SOURCES

Source-integrity rule: list only sources that actually appeared in the browser-search
findings. Never invent a scholar, book, institution, report, article, quotation, URL,
or "current consensus". Every surprising claim used by the writer must be traceable
to the supplied search findings.
""".strip()

    return groq_call(
        GROQ_RESEARCH_MODEL,
        [{"role": "user", "content": synthesis_prompt}],
        max_completion_tokens=4600,
        temperature=0.35,
        attempts=4,
    )


# ---------------------------------------------------------------------------
# Story architecture + script
# ---------------------------------------------------------------------------
STORY_ARCHITECT_PROMPT = """
You are the story architect for Relic Loop, a curiosity-first explainer channel.

Turn research into a story of discovery, not a textbook outline or a list of facts.
The video must have one central curiosity question, escalating reveals, and a satisfying "aha"
that makes the opening question feel obvious in hindsight. The viewer should constantly feel
that one answer has opened the door to a more interesting question.

Structure:
1. COLD OPEN (0-3 seconds): start inside the curiosity with a direct question, surprising fact, contradiction, or strange consequence. Never begin with weather, scenery, a character walking, a boy looking up, or generic historical setup.
2. OPENING PROMISE (3-15 seconds): deepen the mystery and promise a concrete answer or reveal without giving everything away.
3. Give only the context needed to understand why the question matters.
4. ESCALATE: each section must add a new fact, mechanism, comparison, contradiction, consequence, or question.
5. Use a midpoint reversal when the evidence challenges the obvious explanation.
6. Explain the mechanism through clear cause and effect, with concrete examples that can be shown visually.
7. Identify the single core "aha" explanation the viewer has been waiting for.
8. Pay off the opening question directly and distinguish certainty from debate.
9. End with a grounded everyday connection or memorable implication.

RETENTION RULES:
- The first 3 seconds must make the viewer understand the exact curiosity immediately.
- By 15 seconds, introduce the hidden complication or promise of the answer.
- Keep the story moving through "question -> partial answer -> new question -> deeper reveal".
- Every section must change what the viewer thinks they know.
- Use concrete events, decisions, objects, comparisons, consequences, or discoveries instead of
  abstract explanation whenever possible.
- If the topic has a surprising reversal, put it near the middle instead of saving all interest
  for the ending.
- Do not spend long stretches on chronology unless chronology itself explains the mystery.
- Avoid stacking five facts that all prove the same point. Each beat should add a new piece of
  understanding or change the question.
- End by answering the opening question in a way that makes the ordinary thing feel different.

Every section must earn its place. Avoid padding, repeated recaps, fake suspense, and
decorative prose.

Return JSON:
{
  "central_question": "...",
  "hook": "1-3 sentence cold open concept",
  "aha_moment": "the core explanation/reveal the viewer is waiting to understand",
  "story_arc": [
    {"section": 1, "purpose": "...", "reveal": "...", "required_facts": ["..."]}
  ],
  "ending_payoff": "..."
}
""".strip()


def build_story_plan(topic: dict[str, Any], research: str) -> dict[str, Any]:
    base_prompt = (
        STORY_ARCHITECT_PROMPT
        + "\n\nTOPIC:\n"
        + json.dumps(topic, ensure_ascii=False)
        + "\n\nRESEARCH DOSSIER:\n"
        + research
    )

    last_plan: dict[str, Any] | None = None
    for attempt in range(1, 4):
        prompt = base_prompt + """

PREMIUM CHANNEL RETENTION LAYER:
- Treat the episode as a visual investigation, not a lecture.
- The opening 15 seconds must contain a concrete mystery and a specific payoff promise.
- Build 3-5 major reveals, with a meaningful turn or new question every 20-45 seconds.
- Use "because X, but that creates Y" logic to keep answers opening new questions.
- Include at least one memorable comparison, one concrete example, and one consequence viewers can picture.
- Around the midpoint, introduce a reversal, misconception, hidden tradeoff, or unexpected connection.
- The final section must resolve the opening question and explain why the ordinary thing viewers know is actually surprising.
- Avoid fake suspense, repetitive "but there's more" phrasing, and fact dumping.
""".strip()
        if attempt > 1:
            prompt += """
REPAIR PASS: The previous story plan was too thin.
Return at least 10 substantial story_arc sections. Each section must contain:
section (integer), purpose, reveal, and required_facts (2-4 evidence-backed facts).
Do not repeat sections or pad with generic suspense.
""".strip()
        try:
            plan = groq_json(
                GROQ_LIGHT_MODEL,
                [{"role": "user", "content": prompt}],
                max_completion_tokens=4200,
                temperature=0.45,
                attempts=2,
            )
        except Exception as exc:
            print(f"[STORY] architect attempt {attempt} failed: {exc}")
            continue

        last_plan = plan
        arc = plan.get("story_arc")
        if isinstance(arc, list) and len(arc) >= 8 and plan.get("aha_moment"):
            valid = all(
                isinstance(item, dict)
                and item.get("purpose")
                and item.get("reveal")
                and isinstance(item.get("required_facts"), list)
                for item in arc
            )
            if valid:
                print(f"[STORY] validated: {len(arc)} story sections")
                return plan

        print(
            f"[STORY] architect attempt {attempt} returned insufficient structure "
            f"({len(arc) if isinstance(arc, list) else 0} sections)"
        )

    keys = sorted(last_plan.keys()) if isinstance(last_plan, dict) else []
    raise RuntimeError(
        "Story architect could not produce at least 8 complete story sections after 3 attempts. "
        f"Last response keys: {keys}"
    )




SCRIPTWRITER_PROMPT = """
You are the head writer of the curiosity-first YouTube channel Relic Loop.

Write an 8-15 minute narration that answers one irresistible curiosity question. Topics may be everyday life, animals, science, human behavior, technology, culture/society, or history.
The audience is curious teenagers and adults. The language is simple, but the ideas are not childish.

RETENTION-FIRST STORY ARCHITECTURE
1. COLD OPEN — FIRST 0-3 SECONDS
Start immediately with the curiosity. No weather, sunrise, walking, a boy looking up, generic historical scenery, channel greeting, or background setup.
The first sentence must be a direct question, surprising contradiction, striking fact, or impossible-sounding observation clearly connected to the title.
Good pattern: "Have you ever wondered why [ordinary thing] does [strange thing]?"
Other valid patterns: "You see this every day, but here's the strange part..." or "It looks harmless, but [unexpected consequence]."
The opening must be understandable with sound only.

2. OPENING PAYOFF PROMISE — 3-15 SECONDS
Immediately deepen the mystery and promise a concrete answer. Give the viewer a reason to keep watching: a surprising mechanism, hidden reason, counterintuitive explanation, or final reveal. Do not explain everything immediately.

3. RAPID EXPLANATION — 15-60 SECONDS
Establish the simplest piece of the answer, then introduce a second question or surprising consequence. Every answer should create another interesting question.

4. ESCALATING REVEALS
Build from familiar to surprising to deeper mechanism to consequence to final aha. Use specific examples, comparisons, experiments, evidence, and visualizable processes.

5. PAYOFF / LOOPBACK
Return to the original question and make the opening make more sense after the explanation. End cleanly without generic filler.

IMPORTANT RETENTION RULE:
Every 15-30 seconds, introduce a new piece of information, contrast, question, consequence, or visual surprise. Never let the narration sit on the same idea for a long time. The story should feel like a chain of discoveries rather than a list of facts.

VOICE / PERFORMANCE
- Modern, conversational, confident, vivid, energetic.
- Sounds like a sharp human documentary narrator talking directly to the viewer.
- Most sentences should be 8-18 words, with occasional 3-7 word punch lines.
- Use concrete actions, objects, decisions, mechanisms, comparisons, and consequences.
- Use commas, em dashes, and occasional questions for natural TTS rhythm.
- Ask questions when they create forward momentum, then answer or complicate them quickly.
- Vary sentence openings and paragraph rhythm.

DO NOT WRITE LIKE
- a school essay
- an old-fashioned novel
- an encyclopedia
- a travel brochure
- a movie trailer full of fake suspense
- a chronological list of facts with no curiosity thread

Avoid filler such as "the sun was shining," "the water was calm," "little did they know," "in the annals of history," and long scenery descriptions unless the detail changes the story.
Never invent dialogue or inner thoughts and present them as historical facts.
When evidence is uncertain or disputed, say so naturally.
Do not use graphic descriptions.

VISUAL-FIRST WRITING
Write every scene so an animator can understand exactly what should appear on screen.
Whenever the narration introduces a new fact, object, movement, mechanism, comparison, place, reaction, or consequence, make that concrete thing explicit.
If one sentence contains two distinct visual ideas, make the ideas easy to separate into short visual beats.
Do not hide important information inside abstract language.

Return JSON in exactly this shape:
{
  "title_question": "the central curiosity question",
  "era": "short era/civilization label",
  "scenes": [
    {
      "id": 1,
      "narration": "35-100 words of narration for this scene",
      "setting": "simple place description",
      "characters": ["role/person 1", "role/person 2"],
      "action": "what the characters are visibly doing",
      "props": ["prop 1", "prop 2"],
      "mood": "curious / tense / triumphant / confused / etc"
    }
  ],
  "thumbnail": {
    "headline": "2-4 words, curiosity-first, not the full title",
    "subject": "main visual subject",
    "supporting_prop": "one strong prop or symbol",
    "emotion": "exaggerated, instantly readable facial/body emotion matched to the documented event",
    "composition": "left_subject_right_prop / right_subject_left_prop / central_subject",
    "preferred_variant": 1
  }
}

Scene count: 18-38.
Total narration: 1700-2600 words.
Each scene must describe a distinct, useful visual moment. The renderer will split narration into multiple short visual shots, so narration must contain concrete visual information rather than long abstract paragraphs.
""".strip()


def write_script(topic: dict[str, Any], research: str, plan: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        SCRIPTWRITER_PROMPT
        + """

PREMIUM PRODUCTION RULES:
- Write for fast visual storytelling: each scene should contain concrete objects, actions, comparisons, mechanisms, or consequences that can become distinct shots.
- Include a clear midpoint reversal and a final loopback to the opening question.
- Mark natural moments for visual emphasis by using short punchy sentences around important reveals.
- Avoid long stretches where the only visual would be a person talking.
- Make the narration sound energetic and conversational, with deliberate sentence-length variation and occasional short reveal lines.
"""
        + "\n\nTOPIC:\n"
        + json.dumps(topic, ensure_ascii=False)
        + "\n\nSTORY PLAN:\n"
        + json.dumps(plan, ensure_ascii=False)
        + "\n\nRESEARCH DOSSIER:\n"
        + research
    )
    script = groq_json(
        GROQ_WRITER_MODEL,
        [{"role": "user", "content": prompt}],
        max_completion_tokens=7200,
        temperature=0.72,
        attempts=3,
    )

    # Do not kill the run just because the model under-produced. First allow a
    # lenient structural validation, then repair the same story to target length.
    validate_script(script, allow_short=True)
    atomic_write_json(SCRIPT_PATH, script)
    _save_current_json(CURRENT_SCRIPT_PATH, script)
    total_words = _script_word_count(script)
    scene_count = len(script["scenes"])
    avg_target = max(50, min(105, round(2050 / scene_count)))
    min_scene_words = max(40, avg_target - 20)
    max_scene_words = min(120, avg_target + 15)
    needs_repair = total_words < SCRIPT_MIN_WORDS or total_words > SCRIPT_MAX_WORDS
    needs_repair = needs_repair or any(
        not (min_scene_words <= count_words(str(scene.get("narration", ""))) <= max_scene_words)
        for scene in script["scenes"]
    )
    if needs_repair:
        print(f"[SCRIPT] repair pass needed: ~{total_words} words")
        script = _repair_script_length(topic, research, plan, script)

    validate_script(script)
    issues = _script_style_issues(script)
    if issues:
        print(f"[SCRIPT] style polish pass needed: {issues}")
        script = _polish_script_style(topic, research, plan, script, issues)

    validate_script(script)
    return script

def validate_script(script: dict[str, Any], allow_short: bool = False) -> None:
    scenes = script.get("scenes")
    if not isinstance(scenes, list) or not (SCENE_MIN <= len(scenes) <= SCENE_MAX):
        raise RuntimeError(f"Script scene count must be {SCENE_MIN}-{SCENE_MAX}; got {len(scenes) if isinstance(scenes, list) else 0}.")
    total_words = 0
    for i, scene in enumerate(scenes, 1):
        if not scene.get("narration"):
            raise RuntimeError(f"Scene {i} is missing narration.")
        words = count_words(scene["narration"])
        # Initial LLM output may be slightly short; write_script() intentionally
        # runs a repair pass before strict final validation. Allow short,
        # punchy scene narration to reach the repair pass instead of failing
        # before it can rebalance the script.
        minimum_words = 15 if allow_short else 25
        if words < minimum_words or words > 120:
            expected = "15-120 during preliminary validation" if allow_short else "25-120"
            raise RuntimeError(f"Scene {i} narration is {words} words; expected {expected}.")
        for key in ("setting", "characters", "action", "props", "mood"):
            if key not in scene:
                raise RuntimeError(f"Scene {i} missing {key}.")
        total_words += words
    if not allow_short and not (SCRIPT_MIN_WORDS <= total_words <= SCRIPT_MAX_WORDS):
        raise RuntimeError(f"Total script word count {total_words} outside {SCRIPT_MIN_WORDS}-{SCRIPT_MAX_WORDS}.")
    print(f"[SCRIPT] validated: {len(scenes)} scenes, ~{total_words} words")


SCRIPT_STYLE_RED_FLAGS = (
    "the sun was shining",
    "the water was calm",
    "little did they know",
    "in the annals of history",
    "through the mists of time",
    "from the dawn of time",
)


def _script_word_count(script: dict[str, Any]) -> int:
    return sum(count_words(str(scene.get("narration", ""))) for scene in script.get("scenes", []))


def _fit_narration_to_limit(text: str, max_words: int) -> str:
    """Trim an overlong narration at a sentence boundary whenever possible."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    clipped = " ".join(words[:max_words]).strip()
    sentences = re.split(r"(?<=[.!?])\s+", clipped)
    if len(sentences) > 1 and count_words(sentences[-1]) <= 8:
        candidate = " ".join(sentences[:-1]).strip()
        if count_words(candidate) >= 45:
            return candidate
    if not re.search(r"[.!?]$", clipped):
        clipped += "."
    return clipped


def _script_style_issues(script: dict[str, Any]) -> list[str]:
    full_text = " ".join(str(scene.get("narration", "")) for scene in script.get("scenes", []))
    lower = full_text.lower()
    issues = [f"Avoid stale phrase: {phrase}" for phrase in SCRIPT_STYLE_RED_FLAGS if phrase in lower]
    q_count = full_text.count("?")
    if q_count > max(8, len(script.get("scenes", [])) // 2):
        issues.append("Too many rhetorical questions; keep only questions that genuinely advance the explanation.")

    opening = " ".join(str(scene.get("narration", "")) for scene in script.get("scenes", [])[:2]).strip()
    opening_lower = opening.lower()
    if re.match(r"^(hey|hello|hi everyone|welcome|today we're|in this video)", opening_lower):
        issues.append("The opening starts with generic channel/video framing; start inside the interesting situation.")
    if not re.search(r"\b(?:why|how|what|what makes|what causes|what happens|why does|why do|how does|how can)\b", opening_lower):
        issues.append("The opening does not clearly establish the central curiosity question.")
    if not re.search(r"\b(?:because|but|instead|surprisingly|actually|the catch|the strange part|turns out|however)\b", opening_lower):
        issues.append("The opening lacks a clear curiosity turn or contradiction.")
    return issues


def _repair_script_length(topic: dict[str, Any], research: str, plan: dict[str, Any], script: dict[str, Any]) -> dict[str, Any]:
    """
    Expand narration in small batches. Partial progress is written to the
    persistent current-run state after every successful batch.
    """
    current_words = _script_word_count(script)
    scene_count = len(script["scenes"])
    avg_target = max(50, min(105, round(2050 / scene_count)))
    min_scene_words = max(45, avg_target - 20)
    max_scene_words = min(120, avg_target + 15)

    repaired = json.loads(json.dumps(script, ensure_ascii=False))
    atomic_write_json(SCRIPT_PATH, repaired)
    _save_current_json(CURRENT_SCRIPT_PATH, repaired)

    print(
        f"[SCRIPT] batch repair: {current_words} words -> target ~2050 "
        f"({min_scene_words}-{max_scene_words} words/scene)"
    )

    batch_size = 6
    for start in range(0, scene_count, batch_size):
        end = min(scene_count, start + batch_size)
        batch = repaired["scenes"][start:end]
        success = False

        for attempt in range(1, 4):
            prompt = f"""
You are repairing narration for scenes {start + 1}-{end} of a history video.

Expand ONLY the narration text. Do not change scene IDs, setting, characters, action, props, or mood.
Preserve factual meaning and supported claims. Do not invent facts, dialogue, motives, or events.

Each narration should be roughly {min_scene_words}-{max_scene_words} words.
Add useful explanation, consequences, decisions, evidence, examples, and transitions that deepen the central curiosity.
Do NOT add scenery filler, repetition, generic suspense, fake dialogue, or decorative prose.
Keep the modern, conversational storyteller voice.

Return JSON only:
{{"narrations": ["one narration per scene, in order"]}}

TOPIC:
{json.dumps(topic, ensure_ascii=False)}

STORY PLAN:
{json.dumps(plan, ensure_ascii=False)}

SCENES:
{json.dumps(
    [{"id": scene["id"], "narration": scene["narration"], "setting": scene["setting"],
      "characters": scene["characters"], "action": scene["action"], "props": scene["props"],
      "mood": scene["mood"]} for scene in batch],
    ensure_ascii=False,
)}
""".strip()

            try:
                result = groq_json(
                    GROQ_WRITER_MODEL,
                    [{"role": "user", "content": prompt}],
                    max_completion_tokens=2400,
                    temperature=0.50,
                    attempts=3,
                )
                narrations = result.get("narrations")
                if not isinstance(narrations, list) or len(narrations) != len(batch):
                    raise RuntimeError(
                        f"expected {len(batch)} narrations, got "
                        f"{len(narrations) if isinstance(narrations, list) else 0}"
                    )

                batch_valid = True
                for scene, narration in zip(batch, narrations):
                    narration_text = str(narration).strip()
                    words = count_words(narration_text)
                    if words < 45:
                        batch_valid = False
                        raise RuntimeError(
                            f"scene {scene['id']} returned {words} words; minimum is 45"
                        )
                    if words > max_scene_words:
                        narration_text = _fit_narration_to_limit(narration_text, max_scene_words)
                        print(
                            f"[SCRIPT] trimmed scene {scene['id']} from {words} to "
                            f"{count_words(narration_text)} words"
                        )
                    scene["narration"] = narration_text

                success = batch_valid
                if success:
                    break
            except Exception as exc:
                print(f"[SCRIPT] repair batch {start + 1}-{end}, attempt {attempt} failed: {exc}")

        if not success:
            salvaged = True
            for scene in batch:
                words = count_words(str(scene.get("narration", "")))
                if words > max_scene_words:
                    scene["narration"] = _fit_narration_to_limit(
                        str(scene["narration"]),
                        max_scene_words,
                    )
                    salvaged = salvaged and count_words(scene["narration"]) >= 45
            if not salvaged:
                raise RuntimeError(f"Could not repair narration batch {start + 1}-{end} after 3 attempts.")
            print(f"[SCRIPT] salvaged overlong narration in batch {start + 1}-{end} by trimming to safe limits.")

        # Persist progress so a fresh scheduled/manual run resumes here.
        atomic_write_json(SCRIPT_PATH, repaired)
        _save_current_json(CURRENT_SCRIPT_PATH, repaired)
        print(f"[SCRIPT] saved repair progress through scene {end} (~{_script_word_count(repaired)} words)")

    final_words = _script_word_count(repaired)

    # If the batches still landed slightly short overall, top up in the same
    # bounded batches rather than failing because of one underlong scene.
    if final_words < SCRIPT_MIN_WORDS:
        deficit = SCRIPT_MIN_WORDS - final_words
        print(f"[SCRIPT] top-up needed: {deficit} words")
        topup_per_scene = max(8, min(30, int(np.ceil(deficit / scene_count)) + 3))

        for start in range(0, scene_count, batch_size):
            end = min(scene_count, start + batch_size)
            batch = repaired["scenes"][start:end]
            prompt = f"""
Lightly expand the narration for these history-video scenes.

Preserve every fact and the existing wording as much as practical. Add only useful explanation,
context, consequences, or transitions. Do not add scenery, repetition, dialogue, or unsupported claims.

Add roughly {topup_per_scene} useful words to EACH narration. Never take any scene above 120 words.

Return JSON only:
{{"narrations": ["one updated narration per scene, in order"]}}

SCENES:
{json.dumps(
    [{"id": scene["id"], "narration": scene["narration"]} for scene in batch],
    ensure_ascii=False,
)}
""".strip()
            result = groq_json(
                GROQ_WRITER_MODEL,
                [{"role": "user", "content": prompt}],
                max_completion_tokens=1800,
                temperature=0.40,
                attempts=3,
            )
            narrations = result.get("narrations")
            if not isinstance(narrations, list) or len(narrations) != len(batch):
                raise RuntimeError("Top-up returned the wrong number of narrations.")
            for scene, narration in zip(batch, narrations):
                narration_text = str(narration).strip()
                words = count_words(narration_text)
                if words < 45:
                    raise RuntimeError(
                        f"Top-up produced {words} words for scene {scene['id']}; minimum is 45."
                    )
                if words > 120:
                    narration_text = _fit_narration_to_limit(narration_text, 120)
                    print(
                        f"[SCRIPT] trimmed top-up scene {scene['id']} from {words} to "
                        f"{count_words(narration_text)} words"
                    )
                scene["narration"] = narration_text
            atomic_write_json(SCRIPT_PATH, repaired)
            _save_current_json(CURRENT_SCRIPT_PATH, repaired)

        final_words = _script_word_count(repaired)

    if not (SCRIPT_MIN_WORDS <= final_words <= SCRIPT_MAX_WORDS):
        raise RuntimeError(
            f"Script repair finished at {final_words} words; "
            f"expected {SCRIPT_MIN_WORDS}-{SCRIPT_MAX_WORDS}."
        )

    validate_script(repaired)
    print(f"[SCRIPT] repair successful: ~{final_words} words")
    return repaired


def _polish_script_style(topic: dict[str, Any], research: str, plan: dict[str, Any], script: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    prompt = f"""
Polish this finished history script because it triggered these style checks:
{json.dumps(issues, ensure_ascii=False)}

Keep the same factual meaning, central question, scene order, characters, settings, actions, props,
and approximate length. Remove old-fashioned novel phrasing, decorative scenery, filler, and excessive
rhetorical questions. Make it sound like a brilliant modern storyteller speaking directly to a curious
teenager and adults. Do not invent facts, dialogue, motives, or events.

Return JSON only in the same schema as the input.

TOPIC:
{json.dumps(topic, ensure_ascii=False)}

STORY PLAN:
{json.dumps(plan, ensure_ascii=False)}

RESEARCH:
{research}

DRAFT:
{json.dumps(script, ensure_ascii=False)}
""".strip()
    polished = groq_json(
        GROQ_WRITER_MODEL,
        [{"role": "user", "content": prompt}],
        max_completion_tokens=6500,
        temperature=0.55,
        attempts=2,
    )
    validate_script(polished)
    return polished


# ---------------------------------------------------------------------------
# Kokoro TTS
# ---------------------------------------------------------------------------
_kokoro_pipeline = None
_local_worker_process: subprocess.Popen[str] | None = None


def get_kokoro_pipeline():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        try:
            from kokoro import KPipeline
        except Exception as exc:
            raise RuntimeError(
                "Kokoro is not installed. The GitHub workflow should install it "
                "and espeak-ng before running the pipeline."
            ) from exc
        print(f"[TTS] Loading Kokoro voice pipeline for {KOKORO_VOICE}...")
        _kokoro_pipeline = KPipeline(lang_code="a")
    return _kokoro_pipeline


def synthesize_kokoro(text: str, voice: str | None = None) -> np.ndarray:
    pipeline = get_kokoro_pipeline()
    chunks: list[np.ndarray] = []
    selected_voice = voice or KOKORO_VOICE
    generator = pipeline(text, voice=selected_voice, speed=KOKORO_SPEED)
    for _, _, audio in generator:
        if audio is None:
            continue
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size:
            chunks.append(arr)
    if not chunks:
        raise RuntimeError("Kokoro produced no audio.")
    return np.concatenate(chunks)


def generate_voiceovers(script: dict[str, Any]) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(AUDIO_SR * 0.04), dtype=np.float32)
    for idx, scene in enumerate(script["scenes"], 1):
        out = AUDIO_DIR / f"scene_{idx:03d}.wav"
        if out.exists() and out.stat().st_size > 10000:
            print(f"[TTS] Reusing {out.name}")
            continue
        print(f"[TTS] Scene {idx}/{len(script['scenes'])}: generating")
        audio = synthesize_kokoro(scene["narration"])
        sf.write(out, audio, AUDIO_SR)
        if idx != len(script["scenes"]):
            # Add a tiny silence to keep cuts clean without sounding like a gap.
            sf.write(out, np.concatenate([audio, silence]), AUDIO_SR)
    checkpoint("voiceovers_complete")


def voice_test() -> list[Path]:
    """Generate the same hook with several voices for a fair A/B/C comparison."""
    sample = (
        "You see this all the time, but have you ever wondered why it works this way? "
        "At first, the obvious answer seems simple. Then one tiny detail makes that answer fall apart. "
        "And once you see what is really happening, an ordinary part of everyday life suddenly looks very different."
    )
    test_voices = [
        ("current_male", "am_puck"),
        ("female_bella", "af_bella"),
        ("female_nicole", "af_nicole"),
    ]
    outputs: list[Path] = []
    for label, voice in test_voices:
        audio = synthesize_kokoro(sample, voice=voice)
        out = ROOT / f"voice_test_{label}.wav"
        sf.write(out, audio, AUDIO_SR)
        outputs.append(out)
        print(f"[VOICE TEST] wrote {out} ({len(audio) / AUDIO_SR:.1f}s) using {voice}")
    return outputs

def local_image_test() -> None:
    """Generate one CPU image with the isolated OpenVINO worker."""
    test_dir = WORK_DIR / "local_image_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    out = test_dir / "sample_01.jpg"

    prompt = """
Polished 2D historical cartoon, cinematic storybook, expressive believable people,
richly layered environment, crisp ink contours, painterly cel-shaded color, warm natural light.
Ancient African river market at sunrise. A historically grounded merchant speaks with a traveler
beside woven baskets and traded goods. Mud-brick buildings, wooden boats, fabrics, pottery and
trees create depth. No readable text, logos, modern objects, photorealism, 3D CGI, anime, cars,
asphalt or lane markings.
""".strip()

    started = time.time()
    print(f"[LOCAL IMAGE TEST] Loading OpenVINO model: {LOCAL_IMAGE_MODEL}")
    _run_local_image(prompt, out, 9001)

    elapsed = time.time() - started
    checkpoint(
        "local_image_test_complete",
        image_model=LOCAL_IMAGE_MODEL,
        output=str(out.relative_to(ROOT)),
        seconds=round(elapsed, 2),
    )
    print(f"[LOCAL IMAGE TEST] Complete in {elapsed:.1f}s: {out}")

def visual_test() -> None:
    """Generate a few polished sample frames without LLM, TTS, or YouTube."""
    if IMAGE_PROVIDER != "cloudflare":
        raise RuntimeError(
            f"Unsupported IMAGE_PROVIDER={IMAGE_PROVIDER!r}. "
            "The visual test currently requires cloudflare."
        )
    require_secret("CLOUDFLARE_ACCOUNT_ID")
    require_secret("CLOUDFLARE_API_TOKEN")
    test_dir = WORK_DIR / "visual_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        (
            "A powerful ancient African ruler stands before a thriving riverside city at sunrise, "
            "wearing historically grounded royal garments and holding a ceremonial staff; traders, "
            "boats, stone buildings, palms, and a busy market fill the layered background."
        ),
        (
            "Inside an ancient royal workshop, skilled artisans examine a mysterious object on a table "
            "while the ruler and advisors watch closely; detailed period tools, fabrics, architecture, "
            "lamps, shelves, and expressive faces create a believable historical scene."
        ),
        (
            "At dusk, a small group of historical travelers crosses a monumental desert road toward a "
            "distant walled city and temple complex; banners move in the wind, pack animals carry supplies, "
            "and the composition feels like a polished illustrated documentary frame."
        ),
    ]

    previous: Path | None = None
    outputs: list[str] = []
    for idx, description in enumerate(samples, 1):
        out = test_dir / f"sample_{idx:02d}.jpg"
        refs = []
        if previous is not None:
            refs.append(_small_reference(previous, f"visual_test_prev_{idx:02d}"))

        prompt = f"""
{POLISHED_VISUAL_STYLE}

Create a polished historical documentary illustration in 16:9.
The image should feel like a premium hand-illustrated storybook frame rather than clip-art.
Keep human anatomy convincing, faces expressive, environments richly layered, and historical
details coherent. Use the supplied style reference as the visual anchor.

SCENE:
{description}

Continuity:
Preserve the established illustration language and recurring character design from the
reference frames, while creating a distinct composition and setting.

No readable text, captions, subtitles, logos, watermarks, letters, numbers, pseudo-writing,
modern objects, modern roads, asphalt, lane markings, traffic signs, power lines, cars,
photorealism, 3D CGI, anime, stick figures, doodles, or flat geometric art. Use a genuinely
period-appropriate road or path rather than a modern paved roadway. Flags and banners must be
blank or non-readable unless the scene explicitly requires a documented inscription.
""".strip()

        print(f"[VISUAL TEST] Generating sample {idx}/{len(samples)}")
        provider_used = _generate_image_with_fallback(
            prompt,
            out,
            _image_seed(9000, idx - 1),
            refs,
        )
        previous = out
        outputs.append({"path": str(out.relative_to(ROOT)), "provider": provider_used})

    checkpoint(
        "visual_test_complete",
        image_providers=IMAGE_PROVIDER_ORDER,
        primary_image_provider=IMAGE_PROVIDER,
        image_model=CLOUDFLARE_IMAGE_MODEL,
        samples=outputs,
    )
    print("[VISUAL TEST] Complete. Inspect the generated sample images in the workflow artifact.")



# ---------------------------------------------------------------------------
# Polished AI illustration renderer
# ---------------------------------------------------------------------------
def load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


FONT_24 = load_font(FONT_REGULAR, 24)
FONT_28 = load_font(FONT_BOLD, 28)
FONT_42 = load_font(FONT_BOLD, 42)
FONT_64 = load_font(FONT_BOLD, 64)
FONT_88 = load_font(FONT_BOLD, 88)


def contains_any(text: str, words: Iterable[str]) -> bool:
    text = text.lower()
    return any(w.lower() in text for w in words)


def split_visual_beats(narration: str) -> list[str]:
    """
    Split narration by MEANING, not by a word-count target.

    The visual should normally last for one complete sentence or one short phrase/idea.
    A long sentence may be split at a real clause transition, but we never chop narration
    into arbitrary 17-word chunks just to hit a number. This keeps each image synchronized
    with what the narrator is actually saying.
    """
    text = normalize_spaces(narration)
    if not text:
        return [""]

    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    expanded: list[str] = []

    # Split only when a sentence contains multiple independently visualizable ideas.
    clause_break = re.compile(
        r"\s+(?=(?:but|because|so|then|instead|while|which|meaning|that means|"
        r"yet|however|although|when|after|before|once|until|unless|rather than|"
        r"as a result|which is why|this means)\b)",
        re.I,
    )
    punctuation_break = re.compile(r"(?<=[,;:—])\s+")

    for sentence in sentences:
        if count_words(sentence) <= 32:
            expanded.append(sentence)
            continue

        parts = [p.strip() for p in clause_break.split(sentence) if p.strip()]
        if len(parts) == 1:
            # Only use punctuation when it creates a genuine phrase boundary.
            parts = [p.strip() for p in punctuation_break.split(sentence) if p.strip()]

        # If a very long sentence still has no semantic boundary, keep it intact.
        # One image for one sentence is preferable to arbitrary word chopping.
        if len(parts) == 1:
            expanded.append(sentence)
        else:
            bucket = ""
            for part in parts:
                candidate = f"{bucket} {part}".strip()
                if not bucket:
                    bucket = part
                elif count_words(candidate) <= 34:
                    bucket = candidate
                else:
                    expanded.append(bucket)
                    bucket = part
            if bucket:
                expanded.append(bucket)

    # Tiny fragments are visually weak on their own, so attach them to the nearest
    # meaningful phrase. This is the only merging done before the episode cap.
    i = 0
    while i < len(expanded):
        if count_words(expanded[i]) < 5 and len(expanded) > 1:
            if i == 0:
                expanded[1] = f"{expanded[i]} {expanded[1]}".strip()
            else:
                expanded[i - 1] = f"{expanded[i - 1]} {expanded[i]}".strip()
            del expanded[i]
            continue
        i += 1

    return expanded or [text]

POLISHED_VISUAL_STYLE = """
Modern 2D animated-explainer keyframe for Relic Loop.
Crisp clean linework, sharp graphic shapes, polished cel shading, vivid but controlled
colors, strong key/rim lighting, readable silhouettes, clear foreground/midground/background
separation, dynamic perspective, richly designed environments, and premium modern
television-animation finish. Characters or animals are used only when the narration needs them;
otherwise objects, mechanisms, environments, diagrams, and processes are the visual subjects.

This is an explanatory frame, not a generic illustration. Make the specific narrated idea
easier to understand at a glance. Modern topics should use accurate modern objects and
environments; historical topics should use plausible period details.

Use bold shapes, clear cause-and-effect, useful comparisons, cutaways, process views,
before/after states, scale cues, maps, evidence objects, or reaction shots whenever they
clarify the narration.

Do NOT use stick figures, doodles, primitive clip-art, accidental text, captions, subtitles,
logos, watermarks, fake UI screenshots, photorealistic stock-photo style, generic 1960s
educational art, sepia/vintage textbook treatment, or decorative imagery that does not
help explain the beat.
""".strip()

THUMBNAIL_VISUAL_STYLE = """
Modern high-energy YouTube curiosity thumbnail for Relic Loop.
Crisp clean cartoon linework, polished cel shading, vivid contrast, cinematic lighting,
big expressive faces or animals when useful, strong silhouettes, sharp foreground subjects,
simplified high-impact background, rich depth, dramatic perspective, and premium modern
explainer polish.

The thumbnail should instantly communicate ONE QUESTION or mystery: a dominant surprising
thing + a clear contrast/consequence + expressive subjects when helpful. Build curiosity
without generic clickbait.

Do NOT use vintage textbook art, sepia painting, calm formal documentary covers,
photorealistic stock art, 3D CGI, anime, clip-art, cluttered collage, readable text,
letters, numbers, captions, subtitles, logos, watermarks, or invented writing inside the image.
""".strip()


def build_visual_plan(script: dict[str, Any]) -> list[list[str]]:
    """Build one deterministic visual-beat plan shared by render and assembly."""
    plans = [
        split_visual_beats(str(scene.get("narration", "")))
        for scene in script["scenes"]
    ]
    initial = sum(len(beats) for beats in plans)

    def density(text: str) -> float:
        words = max(1, count_words(text))
        signals = len(re.findall(
            r"\b\d{2,4}\b|\b(?:map|route|letter|document|report|record|evidence|"
            r"battle|ship|city|king|queen|decision|discovered|found|arrived|left)\b",
            text,
            re.I,
        ))
        return signals / words

    while sum(len(beats) for beats in plans) > MAX_VISUAL_BEATS_PER_VIDEO:
        candidates = [idx for idx, beats in enumerate(plans) if len(beats) > 1]
        if not candidates:
            break
        idx = min(candidates, key=lambda i: (density(" ".join(plans[i])), i))
        beats = plans[idx]
        pair_idx = min(
            range(len(beats) - 1),
            key=lambda j: (
                density(f"{beats[j]} {beats[j + 1]}"),
                count_words(beats[j]) + count_words(beats[j + 1]),
            ),
        )
        beats[pair_idx:pair_idx + 2] = [
            f"{beats[pair_idx]} {beats[pair_idx + 1]}".strip()
        ]

    final = sum(len(beats) for beats in plans)
    if initial != final:
        print(
            f"[IMAGE PLAN] Reduced visual beats from {initial} to {final} "
            f"to respect the {MAX_VISUAL_BEATS_PER_VIDEO}-image episode cap."
        )
    # Quality guard: remove accidental consecutive duplicate beats after merging.
    for plan_idx, beats in enumerate(plans):
        if len(beats) > 1:
            cleaned = [beats[0]]
            for beat in beats[1:]:
                if normalize_spaces(beat).lower() != normalize_spaces(cleaned[-1]).lower():
                    cleaned.append(beat)
            plans[plan_idx] = cleaned

    final = sum(len(beats) for beats in plans)
    if final > MAX_VISUAL_BEATS_PER_VIDEO:
        raise RuntimeError(
            f"Could not reduce visual plan below {MAX_VISUAL_BEATS_PER_VIDEO} images."
        )
    return plans


def _image_seed(scene_id: int, beat_index: int) -> int:
    return ((scene_id + 1) * 10007 + (beat_index + 1) * 7919) & 0x7FFFFFFF


def _decode_style_reference() -> Path:
    out = WORK_DIR / "visual_style_reference.jpg"
    if out.exists() and out.stat().st_size > 1000:
        return out
    if not STYLE_REFERENCE_B64.exists():
        raise RuntimeError(
            "Missing embedded visual style reference at assets/visual_style_reference.jpg.b64"
        )
    try:
        data = STYLE_REFERENCE_B64.read_bytes()
    except Exception as exc:
        raise RuntimeError("Embedded visual style reference could not be read.") from exc
    if len(data) < 1000 or data[:2] != b"\xff\xd8":
        raise RuntimeError("Embedded visual style reference is not a valid JPEG.")
    out.write_bytes(data)
    return out


def _small_reference(path: Path, label: str) -> Path:
    out = WORK_DIR / f"{label}_reference.jpg"
    if out.exists() and out.stat().st_size > 1000:
        return out
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((448, 448))
        image.save(out, format="JPEG", quality=82, optimize=True)
    return out


class ImageProviderError(RuntimeError):
    def __init__(self, provider: str, message: str, *, disable_for_run: bool = False):
        super().__init__(message)
        self.provider = provider
        self.disable_for_run = disable_for_run


def _save_provider_image_bytes(out_path: Path, raw: bytes, provider: str) -> None:
    if len(raw) < 1000:
        raise ImageProviderError(provider, "Provider returned an empty or tiny image payload.")
    try:
        out_path.write_bytes(raw)
        with Image.open(out_path) as image:
            image.convert("RGB").save(out_path, format="JPEG", quality=92, optimize=True)
    except Exception as exc:
        raise ImageProviderError(provider, f"Provider returned invalid image data: {exc}") from exc


def _huggingface_image(prompt: str, out_path: Path, seed: int) -> None:
    if not HUGGINGFACE_TOKEN:
        raise ImageProviderError("huggingface", "HUGGINGFACE_TOKEN is not configured.", disable_for_run=True)
    try:
        from huggingface_hub import InferenceClient
    except Exception as exc:
        raise ImageProviderError("huggingface", f"huggingface_hub is unavailable: {exc}", disable_for_run=True) from exc
    print(f"[IMAGE] Hugging Face {HUGGINGFACE_IMAGE_MODEL} seed={seed}")
    try:
        client = InferenceClient(api_key=HUGGINGFACE_TOKEN)
        image = client.text_to_image(prompt, model=HUGGINGFACE_IMAGE_MODEL, seed=seed)
        image.save(out_path, format="JPEG", quality=92, optimize=True)
    except Exception as exc:
        status_match = re.search(r"\b(400|401|402|403|404|408|409|429|500|502|503|504)\b", str(exc))
        status = int(status_match.group(1)) if status_match else None
        disable = status in {401, 402, 403, 404, 429}
        raise ImageProviderError("huggingface", f"Hugging Face image generation failed: {str(exc)[:1800]}", disable_for_run=disable) from exc


def _replicate_image(prompt: str, out_path: Path, seed: int) -> None:
    if not REPLICATE_API_TOKEN:
        raise ImageProviderError("replicate", "REPLICATE_API_TOKEN is not configured.", disable_for_run=True)
    url = f"https://api.replicate.com/v1/models/{REPLICATE_IMAGE_MODEL}/predictions"
    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}", "Content-Type": "application/json", "Prefer": "wait=60"}
    payload = {"input": {"prompt": prompt, "seed": seed, "aspect_ratio": "16:9", "output_format": "webp", "output_quality": 90, "safety_tolerance": 2, "prompt_upsampling": False}}
    print(f"[IMAGE] Replicate {REPLICATE_IMAGE_MODEL} seed={seed}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as exc:
        raise ImageProviderError("replicate", f"Replicate request failed: {exc}") from exc
    if response.status_code not in {200, 201, 202}:
        disable = response.status_code in {401, 402, 403, 404, 429}
        raise ImageProviderError("replicate", f"Replicate failed with HTTP {response.status_code}: {response.text[:1600]}", disable_for_run=disable)
    try:
        data = response.json()
    except Exception as exc:
        raise ImageProviderError("replicate", "Replicate returned invalid JSON.") from exc
    status = str(data.get("status", "")).lower()
    prediction_url = data.get("url")
    if prediction_url and status not in {"succeeded", "failed", "canceled"}:
        deadline = time.time() + 300
        while time.time() < deadline:
            time.sleep(5)
            poll = requests.get(prediction_url, headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"}, timeout=60)
            if poll.status_code != 200:
                raise ImageProviderError("replicate", f"Replicate polling returned HTTP {poll.status_code}: {poll.text[:1000]}")
            data = poll.json()
            status = str(data.get("status", "")).lower()
            if status in {"succeeded", "failed", "canceled"}:
                break
    if status != "succeeded":
        error = data.get("error") or "prediction did not succeed"
        disable = "credit" in str(error).lower() or "billing" in str(error).lower() or "payment" in str(error).lower()
        raise ImageProviderError("replicate", f"Replicate prediction ended with status={status}: {str(error)[:1600]}", disable_for_run=disable)
    output = data.get("output")
    output_url = output if isinstance(output, str) else (output[0] if isinstance(output, list) and output and isinstance(output[0], str) else None)
    if not output_url:
        raise ImageProviderError("replicate", f"Replicate returned no output URL: {json.dumps(data)[:1600]}")
    try:
        image_response = requests.get(output_url, timeout=120)
    except requests.RequestException as exc:
        raise ImageProviderError("replicate", f"Replicate image download failed: {exc}") from exc
    if image_response.status_code != 200:
        raise ImageProviderError("replicate", f"Replicate image download failed with HTTP {image_response.status_code}.")
    _save_provider_image_bytes(out_path, image_response.content, "replicate")


def _local_python() -> Path:
    return LOCAL_IMAGE_ENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _ensure_local_image_environment() -> Path:
    """Create the isolated local-image environment on first use in a run."""
    if not LOCAL_IMAGE_WORKER.exists():
        raise ImageProviderError("local", f"Missing local image worker: {LOCAL_IMAGE_WORKER}", disable_for_run=True)
    if not LOCAL_IMAGE_DEPS.exists():
        raise ImageProviderError("local", f"Missing local image dependency file: {LOCAL_IMAGE_DEPS}", disable_for_run=True)

    python_bin = _local_python()
    marker = LOCAL_IMAGE_ENV_DIR / ".ready"
    if python_bin.exists() and marker.exists():
        return python_bin

    print("[LOCAL IMAGE] Creating isolated CPU environment...")
    try:
        if not python_bin.exists():
            subprocess.run(
                [sys.executable, "-m", "venv", "--system-site-packages", str(LOCAL_IMAGE_ENV_DIR)],
                check=True,
                timeout=180,
            )
        subprocess.run(
            [
                str(python_bin), "-m", "pip", "install",
                "--disable-pip-version-check",
                "--upgrade-strategy", "only-if-needed",
                "-r", str(LOCAL_IMAGE_DEPS),
            ],
            check=True,
            timeout=900,
        )
        atomic_write_text(marker, "ready\n")
    except subprocess.SubprocessError as exc:
        raise ImageProviderError(
            "local",
            f"Could not prepare isolated local image environment: {str(exc)[:1800]}",
            disable_for_run=True,
        ) from exc
    return python_bin


def _stop_local_worker() -> None:
    global _local_worker_process
    proc = _local_worker_process
    _local_worker_process = None
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.write('{"cmd":"shutdown"}\n')
            proc.stdin.flush()
    except Exception:
        pass
    try:
        proc.wait(timeout=8)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


import atexit
atexit.register(_stop_local_worker)


def _get_local_worker_process(python_bin: Path) -> subprocess.Popen[str]:
    global _local_worker_process
    if _local_worker_process is not None and _local_worker_process.poll() is None:
        return _local_worker_process

    print("[LOCAL IMAGE] Starting persistent CPU worker (model loads once per run)...")
    child_env = os.environ.copy()
    child_env["MOTIVE_LOCAL_IMAGE_MODEL"] = LOCAL_IMAGE_MODEL
    _local_worker_process = subprocess.Popen(
        [str(python_bin), str(LOCAL_IMAGE_WORKER), "--server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )
    return _local_worker_process


def _run_local_image(prompt: str, out_path: Path, seed: int) -> None:
    python_bin = _ensure_local_image_environment()
    request = {
        "cmd": "generate",
        "model_id": LOCAL_IMAGE_MODEL,
        "prompt": prompt,
        "output_path": str(out_path.resolve()),
        "seed": seed,
        "steps": LOCAL_IMAGE_STEPS,
        "width": LOCAL_IMAGE_WIDTH,
        "height": LOCAL_IMAGE_HEIGHT,
    }
    print(f"[IMAGE] Local OpenVINO {LOCAL_IMAGE_MODEL} seed={seed}")
    proc = _get_local_worker_process(python_bin)
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()

        while True:
            line = proc.stdout.readline()
            if not line:
                raise ImageProviderError("local", "Persistent local image worker exited unexpectedly.")
            line = line.rstrip()
            if line:
                print(line)
            if line.startswith("__LOCAL_OK__"):
                break
            if line.startswith("__LOCAL_ERROR__"):
                raise ImageProviderError("local", line, disable_for_run=False)
    except (BrokenPipeError, OSError, subprocess.SubprocessError) as exc:
        _stop_local_worker()
        raise ImageProviderError("local", f"Local OpenVINO worker communication failed: {str(exc)[:1800]}") from exc

    if not out_path.exists() or out_path.stat().st_size < 10000:
        raise ImageProviderError("local", "Local provider produced no valid image file.")


def _local_image(prompt: str, out_path: Path, seed: int) -> None:
    """CPU-only emergency provider using an isolated OpenVINO LCM worker."""
    _run_local_image(prompt, out_path, seed)


def _generate_image_with_fallback(prompt: str, out_path: Path, seed: int, references: list[Path]) -> str:
    disabled = getattr(_generate_image_with_fallback, "_disabled", set())
    attempted = []
    for provider in IMAGE_PROVIDER_ORDER:
        if provider in disabled:
            continue
        attempted.append(provider)
        try:
            if provider == "cloudflare":
                _cloudflare_image(prompt, out_path, seed, references)
            elif provider == "huggingface":
                _huggingface_image(prompt, out_path, seed)
            elif provider == "replicate":
                _replicate_image(prompt, out_path, seed)
            elif provider == "local":
                _local_image(prompt, out_path, seed)
            else:
                print(f"[IMAGE] Unknown provider {provider!r}, skipping.")
                continue
            print(f"[IMAGE] provider used: {provider}")
            return provider
        except ImageProviderError as exc:
            print(f"[IMAGE FALLBACK] {provider} failed: {exc}")
            if exc.disable_for_run:
                disabled.add(provider)
                setattr(_generate_image_with_fallback, "_disabled", disabled)
                print(f"[IMAGE FALLBACK] disabling {provider} for the rest of this run.")
        except Exception as exc:
            print(f"[IMAGE FALLBACK] {provider} unexpected failure: {exc}")
    raise RuntimeError("All configured image providers failed. Attempted: " + (", ".join(attempted) or "(none)"))

def _cloudflare_safe_retry_prompt(prompt: str) -> str:
    """Build a conservative retry prompt for Cloudflare's aggressive 3030 filter."""
    def section(label: str) -> str:
        match = re.search(
            rf"{re.escape(label)}:\s*(.*?)(?=\n[A-Z][A-Z /_-]+:|$)",
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return normalize_spaces(match.group(1)) if match else ""

    beat = section("NARRATION BEAT")
    setting = section("SETTING")
    action = section("VISIBLE ACTION")
    props = section("PROPS / SYMBOLS")
    device = section("VISUAL DEVICE")

    return f"""
Modern 2D animated educational explainer frame.
Crisp clean linework, sharp graphic shapes, polished cel shading, clear lighting,
strong silhouettes, detailed environment, premium television-animation finish.
Create a safe, factual, object/process-first illustration.

NARRATED IDEA:
{beat}

SETTING:
{setting}

VISIBLE ACTION:
{action}

IMPORTANT OBJECTS:
{props}

EXPLANATORY DEVICE:
{device}

Show only the concrete subject and process required by the narrated idea.
No portraits, no glamour, no body-focused framing, no suggestive content, no nudity,
no revealing clothing, no sexualized posing, no unrelated people, no readable text,
logos, watermarks, captions, subtitles, or decorative subjects.
""".strip()


def _cloudflare_image(prompt: str, out_path: Path, seed: int, references: list[Path]) -> None:
    if "cloudflare" not in IMAGE_PROVIDER_ORDER:
        raise ImageProviderError("cloudflare", "Cloudflare is not enabled in IMAGE_PROVIDERS.", disable_for_run=True)
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
        raise ImageProviderError(
            "cloudflare",
            "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required for polished image generation.",
            disable_for_run=True,
        )

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{CLOUDFLARE_IMAGE_MODEL}"
    )
    data = {
        "prompt": prompt,
        "width": str(IMAGE_W),
        "height": str(IMAGE_H),
        "seed": str(seed),
    }
    files: dict[str, tuple[str, bytes, str]] = {}
    for idx, ref in enumerate(references[:4]):
        files[f"input_image_{idx}"] = (
            ref.name,
            ref.read_bytes(),
            "image/jpeg",
        )

    print(f"[IMAGE] Cloudflare {CLOUDFLARE_IMAGE_MODEL} seed={seed}")
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
            data=data,
            files=files or None,
            timeout=180,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Cloudflare image request failed: {exc}") from exc

    if response.status_code != 200:
        message = response.text[:2500]

        # FLUX.2 Klein can return 3030 for benign prompts because its hosted
        # content filter is intentionally conservative. Retry once with a
        # stripped, object/process-first prompt before abandoning the provider.
        # This retry never calls Groq.
        if response.status_code == 400 and (
            "3030" in message or "output has been flagged" in message.lower()
            or "contains NSFW content" in message.lower()
        ):
            safe_prompt = _cloudflare_safe_retry_prompt(prompt)
            print("[IMAGE] Cloudflare 3030 filter hit; retrying with sanitized visual prompt.")
            retry_data = {
                "prompt": safe_prompt,
                "width": str(IMAGE_W),
                "height": str(IMAGE_H),
                "seed": str((seed + 1) & 0x7FFFFFFF),
            }
            try:
                retry_response = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
                    data=retry_data,
                    files=None,
                    timeout=180,
                )
            except requests.RequestException as exc:
                raise ImageProviderError("cloudflare", f"Cloudflare safe-prompt retry failed: {exc}") from exc

            if retry_response.status_code == 200:
                try:
                    retry_payload = retry_response.json()
                    retry_result = retry_payload.get("result")
                    retry_b64 = retry_result.get("image") if isinstance(retry_result, dict) else None
                    if isinstance(retry_b64, str) and retry_b64.strip():
                        out_path.write_bytes(base64.b64decode(retry_b64))
                        with Image.open(out_path) as image:
                            image.convert("RGB").save(out_path, format="JPEG", quality=92, optimize=True)
                        return
                except Exception as exc:
                    print(f"[IMAGE] Cloudflare safe-prompt retry returned unusable image: {exc}")

            retry_message = retry_response.text[:1800]
            if retry_response.status_code == 429:
                raise ImageProviderError(
                    "cloudflare",
                    f"Cloudflare safe-prompt retry hit HTTP 429: {retry_message}",
                    disable_for_run=True,
                )
            raise ImageProviderError(
                "cloudflare",
                f"Cloudflare original 3030 filter rejection and sanitized retry failed with HTTP "
                f"{retry_response.status_code}: {retry_message}",
            )

        disable = response.status_code in {401, 403, 429}
        raise ImageProviderError(
            "cloudflare",
            f"Cloudflare image generation failed with HTTP {response.status_code}: {message}",
            disable_for_run=disable,
        )

    try:
        payload = response.json()
        result = payload.get("result")
        if isinstance(result, dict):
            image_b64 = result.get("image")
        else:
            image_b64 = None
    except Exception as exc:
        raise ImageProviderError("cloudflare", "Cloudflare image response was not valid JSON.") from exc

    if not isinstance(image_b64, str) or not image_b64.strip():
        raise ImageProviderError("cloudflare", "Cloudflare image response did not contain result.image.")

    try:
        out_path.write_bytes(base64.b64decode(image_b64))
    except Exception as exc:
        raise ImageProviderError("cloudflare", "Cloudflare returned invalid base64 image data.") from exc

    with Image.open(out_path) as image:
        image.convert("RGB").save(out_path, format="JPEG", quality=92, optimize=True)


def visual_device_hint(beat_text: str) -> str:
    lower = beat_text.lower()
    if re.search(r"\b(?:because|causes|works by|allows|prevents|helps|means|so that|leads to)\b", lower):
        return "MECHANISM / CAUSE-EFFECT: show a visible process, cutaway, arrows, flow, or step-by-step explanation."
    if re.search(r"\b(?:before|after|used to|now|instead|rather than|changed from|became)\b", lower):
        return "BEFORE / AFTER: show a clear contrast between two states, designs, behaviors, or situations."
    if re.search(r"\b(?:larger|smaller|twice|half|percent|million|thousand|only|tiny|huge)\b", lower):
        return "SCALE / COMPARISON: use physical objects, groups, silhouettes, or side-by-side scale cues without readable labels."
    if re.search(r"\b(?:map|route|crossed|traveled|sailed|marched|arrived|distance|border|river|coast|road|spread|moved)\b", lower):
        return "GEOGRAPHY / MOVEMENT: use a map-like or overhead composition with clear positions and movement."
    if re.search(r"\b(?:brain|nerve|cell|blood|lung|heart|gill|bone|muscle|inside|under the skin|anatom)\b", lower):
        return "BIOLOGICAL CUTAWAY: show a clean explanatory cross-section of the relevant internal mechanism."
    if re.search(r"\b(?:letter|document|report|record|diary|decree|note|inscription|photograph|evidence|testimony|study|experiment)\b", lower):
        return "EVIDENCE CLOSE-UP: make the real object, data, or source visually prominent and explain why it matters."
    if re.search(r"\b(?:decided|ordered|refused|agreed|claimed|argued|revealed|discovered|found|realized|learned)\b", lower):
        return "REACTION / REVEAL: use a strong reaction or discovery moment with a visible clue or consequence."
    if re.search(r"\b(?:how many|how long|steps|first|second|then|finally|process|build|made|formed)\b", lower):
        return "PROCESS SEQUENCE: show a physical transformation or a distinct step in the process."
    return "CINEMATIC EXPLANATION: choose the clearest concrete scene, object, person, animal, behavior, or action that makes the beat understandable.";


def make_visual_prompt(
    scene: dict[str, Any],
    era: str,
    beat_text: str,
    beat_index: int,
    beat_count: int,
    has_previous_reference: bool,
    topic_category: str = "general",
    central_question: str = "",
) -> str:
    def visual_shot_type(text: str, index: int) -> str:
        lower = text.lower()
        if re.search(r"\b(?:because|causes|works by|allows|prevents|helps|means|leads to)\b", lower):
            return "close explanatory mechanism shot or cutaway with the cause visibly leading to the effect"
        if re.search(r"\b(?:before|after|instead|rather than|changed from|became)\b", lower):
            return "clear before/after comparison with both states visually legible"
        if re.search(r"\b(?:brain|nerve|cell|blood|lung|heart|gill|bone|muscle|inside|anatom)\b", lower):
            return "clean biological or mechanical cutaway with the relevant internal part emphasized"
        if re.search(r"\b(?:map|route|crossed|traveled|sailed|marched|arrived|distance|border|river|coast|road|spread|moved)\b", lower):
            return "high-angle geographic or movement shot with a clear route or positional relationship"
        if re.search(r"\b(?:document|record|letter|report|evidence|study|experiment|photograph)\b", lower):
            return "tight evidence close-up with the key object/data and a human interaction anchoring why it matters"
        if re.search(r"\b(?:decided|ordered|refused|agreed|revealed|discovered|found|realized|learned)\b", lower):
            return "dynamic reaction/reveal shot with a strong gesture, eye-line, clue, or consequence"
        if re.search(r"\b(?:first|second|then|finally|process|build|made|formed|steps)\b", lower):
            return "process-focused explanatory composition showing a distinct stage or transformation"
        fallbacks = [
            "wide cinematic establishing shot with layered depth and one dominant subject",
            "medium interaction shot with readable expressions and specific gestures",
            "over-the-shoulder shot focused on the exact object, behavior, or consequence being explained",
            "low-angle reveal or close reaction shot with a strong foreground focal point",
        ]
        return fallbacks[index % len(fallbacks)]

    shot = visual_shot_type(beat_text, beat_index)
    raw_chars = [normalize_spaces(str(x)) for x in (scene.get("characters") or []) if normalize_spaces(str(x))]
    raw_props = [normalize_spaces(str(x)) for x in (scene.get("props") or []) if normalize_spaces(str(x))]
    chars = ", ".join(raw_chars[:4])
    props = ", ".join(raw_props[:4])

    # People are optional visual subjects, not a default. The old fallback of
    # "historical people" was causing science/everyday-life beats to become
    # unrelated character portraits.
    people_words = (
        "person", "people", "human", "cook", "chef", "scientist", "researcher",
        "farmer", "worker", "doctor", "child", "man", "woman", "family", "crowd",
        "customer", "driver", "engineer", "inventor", "author", "ruler", "soldier",
        "king", "queen", "emperor", "empress"
    )
    beat_requires_people = contains_any(beat_text, people_words)
    if raw_chars and beat_requires_people:
        people_instruction = (
            "PEOPLE ALLOWED: only the named people relevant to the narration. "
            "Keep them secondary unless the narration is about their action or reaction."
        )
    else:
        people_instruction = (
            "OBJECT / PROCESS FIRST: do not add human figures or portraits. "
            "Use the exact object, animal, environment, mechanism, evidence, or process "
            "named by the narration as the dominant subject."
        )

    setting = normalize_spaces(str(scene.get("setting", ""))) or "a setting that directly matches the narration beat"
    action = normalize_spaces(str(scene.get("action", ""))) or "the exact physical action described in the narration beat"
    props = props or "only the specific physical objects required by the narration beat"
    device = visual_device_hint(beat_text)

    continuity = (
        "A previous generated frame is supplied as a reference. Preserve only relevant "
        "visual continuity from it while creating a genuinely new shot with new visual "
        "information. Never copy an unrelated subject into this frame."
        if has_previous_reference
        else
        "No unrelated reference subject is being carried into this frame."
    )

    return f"""
{POLISHED_VISUAL_STYLE}

TOPIC CATEGORY:
{topic_category}

CENTRAL QUESTION:
{central_question or "(not supplied)"}

SETTING / CONTEXT:
{era}

SETTING:
{setting}

CHARACTERS:
{chars or "none required"}

VISIBLE ACTION:
{action}

PEOPLE RULE:
{people_instruction}

PROPS / SYMBOLS:
{props}

MOOD:
{scene.get("mood", "curious")}

NARRATION BEAT:
{beat_text}

VISUAL DEVICE:
{device}

SHOT DIRECTION:
{shot}. The frame must communicate the narration beat at a glance. Favor animation-keyframe staging:
clear silhouette, one dominant focal action, one strong secondary story clue, exaggerated readable
expressions, and visible cause-and-effect. Prefer specific physical details, accurate materials, clothing, tools, architecture, terrain, technology, anatomy, and everyday objects for the setting.
When the beat introduces a new fact, location, date, object, movement, or consequence, make that
new information the focal point. Avoid generic "people standing around" compositions.

CONTINUITY:
{continuity}

NON-NEGOTIABLE CONTENT RULE:
The NARRATION BEAT is the source of truth for the image. The dominant visual must directly
depict the exact thing being explained in that beat. Do not substitute generic attractive
characters, portraits, fashion imagery, unrelated historical scenes, or decorative subjects.
If the beat describes a scientific mechanism, prefer a clean object/cutaway/process view.
If the beat describes an animal, show that animal and its behavior. If it describes an object,
show that object clearly and at useful scale. If it describes a place or event, show that place
or event. People appear only when the narration actually requires them.

SAFETY / CLEAN VISUALS:
No nudity, underwear-focused imagery, sexualized posing, glamour portraits, fetish styling,
or body-focused compositions. Keep clothing ordinary and age-appropriate whenever people are
actually needed.

Create a finished, polished illustration. No readable text, lettering, pseudo-writing, logos,
watermarks, or accidental modern signage. Documents, screens, labels, or diagrams may be shown as detailed objects, but do not invent readable text unless the narration explicitly requires documented text.
""".strip()


def render_scenes(script: dict[str, Any]) -> None:
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    visual_plan = build_visual_plan(script)
    total_beats = sum(len(beats) for beats in visual_plan)

    previous_image: Path | None = None
    previous_characters: set[str] = set()
    topic_category = str(script.get("topic_category", "general"))
    central_question = str(script.get("central_question", ""))

    for idx, scene in enumerate(script["scenes"], 1):
        beats = visual_plan[idx - 1]
        current_characters = {
            normalize_spaces(str(x)).lower()
            for x in (scene.get("characters") or [])
            if normalize_spaces(str(x))
        }
        for beat_idx, beat_text in enumerate(beats, 1):
            out = SCENE_DIR / f"scene_{idx:03d}_beat_{beat_idx:02d}.jpg"
            if out.exists() and out.stat().st_size > 10000:
                print(f"[IMAGE] Reusing {out.name}")
                previous_image = out
                continue

            refs = []
            use_previous = previous_image is not None and (
                beat_idx > 1 and bool(current_characters & previous_characters)
            )
            if use_previous:
                refs.append(_small_reference(previous_image, f"prev_{idx:03d}_{beat_idx:02d}"))

            prompt = make_visual_prompt(
                scene,
                script.get("era", "History"),
                beat_text,
                beat_idx - 1,
                len(beats),
                has_previous_reference=use_previous,
                topic_category=topic_category,
                central_question=central_question,
            )
            print(f"[IMAGE] Scene {idx}/{len(script['scenes'])} beat {beat_idx}/{len(beats)}")
            try:
                provider_used = _generate_image_with_fallback(
                    prompt,
                    out,
                    _image_seed(idx, beat_idx - 1),
                    refs,
                )
            except RuntimeError as exc:
                # Never throw away an otherwise complete episode because one
                # visual beat cannot be generated. Reuse the most recent valid
                # frame as a deterministic last-resort visual; this consumes no
                # API tokens and lets the pipeline finish/upload.
                if previous_image is not None and previous_image.exists():
                    with Image.open(previous_image) as fallback:
                        ImageOps.fit(
                            fallback.convert("RGB"),
                            (IMAGE_W, IMAGE_H),
                            method=Image.Resampling.LANCZOS,
                        ).save(out, format="JPEG", quality=92, optimize=True)
                    provider_used = "reused_previous_frame"
                    print(f"[IMAGE FALLBACK] all providers failed for {out.name}; reused previous frame: {exc}")
                else:
                    raise
            previous_image = out

        previous_characters = current_characters

    checkpoint(
        "scenes_complete",
        visual_beats=total_beats,
        image_provider=IMAGE_PROVIDER,
        image_model=CLOUDFLARE_IMAGE_MODEL,
    )



# ---------------------------------------------------------------------------
# SEO
# ---------------------------------------------------------------------------
SEO_PROMPT = """
You are the YouTube packaging editor for Relic Loop, a curiosity-first explainer channel.

Create metadata as a TITLE + THUMBNAIL pair for a video answering one specific question
about something viewers can recognize, experience, or easily imagine.

Rules:
- Primary title under 70 characters; natural curiosity, no fake claims.
- Two alternate titles.
- Description: first two lines explain the question and why it matters; then a spoiler-light
  explanation and a Sources section using only supplied sources.
- Tags: 12-15 relevant terms.
- Thumbnail headline: 2-4 punchy words that add a second curiosity cue rather than repeating the title.
- Title and thumbnail must work together: the title asks or implies the question; the image makes
  the unanswered part visually obvious.
- Avoid generic clickbait such as "YOU WON'T BELIEVE" and empty listicle phrasing.

Return JSON only:
{
  "title": "...",
  "alternate_titles": ["...", "..."],
  "description": "...",
  "tags": ["..."],
  "primary_keywords": ["..."],
  "thumbnail_headline": "..."
}
""".strip()


SEO_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 70},
        "alternate_titles": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        },
        "description": {"type": "string", "minLength": 80},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 12,
            "maxItems": 15,
        },
        "primary_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 8,
        },
        "thumbnail_headline": {"type": "string", "minLength": 2, "maxLength": 40},
    },
    "required": [
        "title",
        "alternate_titles",
        "description",
        "tags",
        "primary_keywords",
        "thumbnail_headline",
    ],
    "additionalProperties": False,
}


def build_seo(topic: dict[str, Any], script: dict[str, Any], research: str) -> dict[str, Any]:
    prompt = (
        SEO_PROMPT
        + """

PREMIUM PACKAGING RULES:
- Title and thumbnail must create complementary curiosity rather than repeat each other.
- Prefer a specific familiar mystery over a generic topic title.
- Thumbnail headline should be 2-4 words and add curiosity, not restate the title.
- Description should open with the central mystery and naturally include the key search phrase.
"""
        + "\n\nCENTRAL QUESTION:\n"
        + topic["question"]
        + "\n\nSCRIPT:\n"
        + "\n\n".join(scene["narration"] for scene in script["scenes"])
        + "\n\nRESEARCH SOURCES / NOTES:\n"
        + research[-12000:]
        + """

IMPORTANT JSON RULES:
- "description" must be one plain string value. Do not put Tags, Sources, or any other JSON key inside it.
- "tags" must be a JSON array of strings, separate from description.
- "alternate_titles" must contain exactly two strings.
- Return only the JSON object. No markdown fences.
"""
    )

    if client is None:
        raise RuntimeError("GROQ_API_KEY is required for SEO generation.")

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=GROQ_LIGHT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=3200,
                temperature=0.45,
                reasoning_effort="low",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "motive_unknown_seo",
                        "strict": True,
                        "schema": SEO_JSON_SCHEMA,
                    },
                },
            )
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise RuntimeError("Strict SEO JSON call returned an empty response.")
            seo = json.loads(raw)
            if not seo.get("title") or not seo.get("description") or not seo.get("tags"):
                raise RuntimeError("SEO output is incomplete.")
            break
        except Exception as exc:
            last_error = exc
            if _is_non_retryable(exc):
                raise RuntimeError(f"SEO request was rejected: {exc}") from exc
            if attempt >= 3:
                raise RuntimeError(f"SEO generation failed after 3 attempts: {exc}") from exc
            wait = _retry_wait(exc, attempt, base=3.0)
            print(f"[SEO RETRY] attempt {attempt}/3: {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)

    if not isinstance(seo, dict):
        raise RuntimeError("SEO generation returned an invalid object.")

    seo["title"] = normalize_spaces(str(seo["title"]))[:70]
    seo["alternate_titles"] = [
        normalize_spaces(str(x))[:100]
        for x in seo.get("alternate_titles", [])[:2]
    ]
    seo["description"] = str(seo["description"]).strip()
    seo["tags"] = [
        normalize_spaces(str(x))
        for x in seo.get("tags", [])
        if normalize_spaces(str(x))
    ][:15]
    seo["primary_keywords"] = [
        normalize_spaces(str(x))
        for x in seo.get("primary_keywords", [])
        if normalize_spaces(str(x))
    ]
    seo["thumbnail_headline"] = normalize_spaces(
        str(seo.get("thumbnail_headline", "History Mystery"))
    )[:40]

    if len(seo["alternate_titles"]) != 2 or not (12 <= len(seo["tags"]) <= 15):
        raise RuntimeError("SEO output did not satisfy title/tag requirements.")

    return seo



def package_quality_issues(topic: dict[str, Any], script: dict[str, Any], seo: dict[str, Any]) -> list[str]:
    """Final low-cost packaging gate; warnings do not make the pipeline brittle."""
    issues: list[str] = []
    question = normalize_spaces(str(topic.get("question", "")))
    title = normalize_spaces(str(seo.get("title", "")))
    headline = normalize_spaces(str(seo.get("thumbnail_headline", "")))
    if len(title) < 18:
        issues.append("Title is unusually short.")
    if not (re.search(r"\?", title) or re.search(r"\b(?:why|how|what|when|do|does|can|is|are)\b", title.lower())):
        issues.append("Title does not clearly imply a curiosity question.")
    if len(headline.split()) > 4:
        issues.append("Thumbnail headline exceeds four words.")
    if title.lower() == question.lower():
        issues.append("Title copies the internal question too literally.")
    overlap = set(re.findall(r"[a-z]{4,}", title.lower())) & set(re.findall(r"[a-z]{4,}", headline.lower()))
    if len(overlap) >= 3:
        issues.append("Title and thumbnail headline repeat too many words.")
    return issues


# ---------------------------------------------------------------------------
# AI thumbnail
# ---------------------------------------------------------------------------
def make_thumbnail(script: dict[str, Any], title: str, headline_override: str | None = None) -> Path:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    final = OUTPUT_DIR / "thumbnail.jpg"
    cached_ai = THUMB_DIR / "thumbnail_ai.jpg"

    thumb = script.get("thumbnail", {}) or {}
    subject = str(thumb.get("subject", "historical figure")).strip()
    prop = str(thumb.get("supporting_prop", "important historical object")).strip()
    emotion = str(thumb.get("emotion", "surprised and curious")).strip()
    composition = str(thumb.get("composition", "left_subject_right_prop")).strip()
    era = str(script.get("era", "Modern day")).strip()

    side_note = {
        "left_subject_right_prop": "Place the main subject prominently on the left and the important object or symbol on the right.",
        "right_subject_left_prop": "Place the main subject prominently on the right and the important object or symbol on the left.",
        "central_subject": "Place the main subject prominently near the center with the important object or symbol clearly visible beside them.",
    }.get(composition, "Use a strong asymmetrical YouTube thumbnail composition with a clear focal subject.")

    style_ref = _small_reference(_decode_style_reference(), "thumbnail_style")

    prompt = f"""
{THUMBNAIL_VISUAL_STYLE}

Create a polished 16:9 YouTube thumbnail illustration for a Relic Loop curiosity video.

ERA:
{era}

MAIN SUBJECT:
{subject}

IMPORTANT OBJECT / SYMBOL:
{prop}

EXPRESSION / BODY LANGUAGE:
{emotion}

COMPOSITION:
{side_note}

Make the main subject and reaction faces LARGE. Push the facial expressions and body language:
shock, disbelief, panic, confusion, amazement, or intense curiosity as appropriate to the facts.
Use one dominant mystery and one obvious visual consequence. Keep the composition simple enough
to read instantly at thumbnail size. Avoid generic portraits or passive people standing still.

No readable text, letters, numbers, pseudo-writing, captions, subtitles, logos, watermarks,
modern infrastructure, modern clothing, cars, asphalt lane markings, or other anachronisms.
""".strip()

    if not cached_ai.exists() or cached_ai.stat().st_size < 10000:
        try:
            _generate_image_with_fallback(prompt, cached_ai, 71003, [style_ref])
        except RuntimeError as exc:
            # Thumbnail failure must not kill a finished video. Prefer a real
            # generated scene frame over a synthetic placeholder.
            scene_candidates = sorted(
                SCENE_DIR.glob("scene_*.jpg"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True,
            )
            if scene_candidates:
                cached_ai.write_bytes(scene_candidates[0].read_bytes())
                print(f"[THUMBNAIL] AI thumbnail failed; using generated scene frame: {exc}")
            else:
                image = Image.new("RGB", (IMAGE_W, IMAGE_H), PAPER)
                draw_fallback = ImageDraw.Draw(image)
                draw_fallback.rectangle([0, 0, IMAGE_W, IMAGE_H], fill=INK)
                draw_fallback.ellipse([120, 100, 560, 540], fill=BLUE)
                draw_fallback.ellipse([650, 160, 1080, 590], fill=GOLD)
                cached_ai.parent.mkdir(parents=True, exist_ok=True)
                image.save(cached_ai, format="JPEG", quality=92, optimize=True)
                print(f"[THUMBNAIL] AI thumbnail failed; using safe graphic fallback: {exc}")

    with Image.open(cached_ai) as base:
        image = base.convert("RGB").resize((1280, 720))

    draw = ImageDraw.Draw(image)
    headline = normalize_spaces(
        str(headline_override or thumb.get("headline") or title or "HISTORY MYSTERY")
    ).upper()
    headline = headline[:36]

    font = FONT_88
    max_width = 720
    words = headline.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    lines = lines[:3]

    x = 45 if "right" not in composition else 720
    y = 45
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=3)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.rounded_rectangle(
            [x - 12, y - 12, min(1260, x + w + 24), y + h + 24],
            radius=18,
            fill=BLACK,
        )
        draw.text(
            (x, y),
            line,
            font=font,
            fill=GOLD if not lines or line == lines[0] else WHITE,
            stroke_width=3,
            stroke_fill=BLACK,
        )
        y += h + 24

    # Keep the thumbnail focused on one mystery; the title supplies context.\n\n    image.save(final, format="JPEG", quality=94, optimize=True)
    print(f"[THUMBNAIL] AI thumbnail ready: {final}")
    return final



def _make_emphasis_filter(scene_durations: list[float]) -> str:
    """Create sparse reveal cards without turning the video into subtitles."""
    if not ENABLE_EMPHASIS_CARDS or not scene_durations:
        return ""
    total = sum(scene_durations)
    anchors = [
        (min(5.5, max(2.0, total * 0.015)), "WHY?"),
        (total * 0.25, "BUT WHY?"),
        (total * 0.50, "THE TWIST"),
        (total * 0.73, "THE REAL REASON"),
        (max(0.0, total - min(8.0, total * 0.08)), "SO THAT'S WHY"),
    ]
    filters: list[str] = []
    for start, phrase in anchors:
        end = min(total, start + 1.35)
        escaped = phrase.replace("'", "\\'")
        filters.append(
            "drawtext="
            f"fontfile={FONT_BOLD}:text='{escaped}':"
            "fontcolor=white:fontsize=54:"
            "box=1:boxcolor=black@0.72:boxborderw=18:"
            "x=(w-text_w)/2:y=h-150:"
            f"enable='between(t\\,{start:.2f}\\,{end:.2f})'"
        )
    return ",".join(filters)


def _make_sound_design_track(total_duration: float, scene_durations: list[float], beat_durations: list[float]) -> Path | None:
    """Generate subtle original transition/reveal SFX so no extra asset is required."""
    if not ENABLE_SOUND_DESIGN or total_duration <= 0:
        return None

    sample_count = int(total_duration * AUDIO_SR) + 1
    track = np.zeros(sample_count, dtype=np.float32)

    def add_whoosh(t0: float, length: float = 0.18) -> None:
        start = int(max(0.0, t0) * AUDIO_SR)
        n = min(int(length * AUDIO_SR), sample_count - start)
        if n <= 0:
            return
        tt = np.arange(n, dtype=np.float32) / AUDIO_SR
        freq = 180.0 + 1100.0 * (tt / max(length, 0.001))
        phase = 2.0 * np.pi * np.cumsum(freq) / AUDIO_SR
        env = np.sin(np.pi * np.clip(tt / max(length, 0.001), 0, 1)) ** 2
        track[start:start+n] += 0.055 * np.sin(phase) * env

    def add_impact(t0: float, length: float = 0.16) -> None:
        start = int(max(0.0, t0) * AUDIO_SR)
        n = min(int(length * AUDIO_SR), sample_count - start)
        if n <= 0:
            return
        tt = np.arange(n, dtype=np.float32) / AUDIO_SR
        env = np.exp(-18.0 * tt)
        track[start:start+n] += 0.045 * np.sin(2.0 * np.pi * 115.0 * tt) * env

    elapsed = 0.0
    for i, dur in enumerate(scene_durations):
        if i > 0:
            add_whoosh(elapsed)
        elapsed += dur

    elapsed = 0.0
    for i, dur in enumerate(beat_durations):
        if i > 0 and i % 7 == 0:
            add_impact(elapsed)
        elapsed += dur

    path = WORK_DIR / "sound_design.wav"
    sf.write(path, np.clip(track, -0.18, 0.18), AUDIO_SR)
    return path

# ---------------------------------------------------------------------------
# FFmpeg / video assembly
# ---------------------------------------------------------------------------
def run_cmd(args: list[str], label: str) -> None:
    print(f"[FFMPEG] {label}")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}).\n"
            f"STDERR:\n{result.stderr[-4000:]}"
        )


def audio_duration(path: Path) -> float:
    data, sr = sf.read(path, always_2d=False)
    length = len(data) / sr
    return float(length)


def build_concat_file(paths: list[Path], output: Path, durations: list[float] | None = None) -> None:
    lines: list[str] = []
    for idx, path in enumerate(paths):
        lines.append(f"file '{path.resolve().as_posix()}'")
        if durations is not None:
            lines.append(f"duration {max(0.25, durations[idx]):.4f}")
    if durations is not None and paths:
        lines.append(f"file '{paths[-1].resolve().as_posix()}'")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _visual_beat_durations(
    narration: str,
    total_audio_duration: float,
    beats: list[str] | None = None,
) -> list[float]:
    beats = beats or split_visual_beats(narration)
    if len(beats) == 1:
        return [total_audio_duration]

    weights = [max(1, count_words(x)) for x in beats]
    total_weight = sum(weights)
    raw = [total_audio_duration * w / total_weight for w in weights]

    # Keep very short flashes readable, then renormalize to the exact audio duration.
    adjusted = [max(VISUAL_BEAT_MIN_DURATION, d) for d in raw]
    scale = total_audio_duration / sum(adjusted)
    return [d * scale for d in adjusted]


def build_video(script: dict[str, Any], out_path: Path) -> float:
    scenes = script["scenes"]
    audio_paths = [AUDIO_DIR / f"scene_{i:03d}.wav" for i in range(1, len(scenes) + 1)]
    for path in audio_paths:
        if not path.exists():
            raise RuntimeError(f"Missing narration asset: {path}")

    all_images: list[Path] = []
    all_durations: list[float] = []
    scene_audio_durations: list[float] = []
    visual_plan = build_visual_plan(script)

    for idx, scene in enumerate(scenes, 1):
        audio_path = AUDIO_DIR / f"scene_{idx:03d}.wav"
        duration = audio_duration(audio_path)
        scene_audio_durations.append(duration)
        beats = visual_plan[idx - 1]
        beat_paths = [
            SCENE_DIR / f"scene_{idx:03d}_beat_{beat_idx:02d}.jpg"
            for beat_idx in range(1, len(beats) + 1)
        ]
        for path in beat_paths:
            if not path.exists():
                raise RuntimeError(f"Missing visual beat asset: {path}")
        beat_durations = _visual_beat_durations(str(scene.get("narration", "")), duration, beats)
        all_images.extend(beat_paths)
        all_durations.extend(beat_durations)

    total = sum(scene_audio_durations)
    if not all_images:
        raise RuntimeError("No visual beat images were generated.")

    video_list = WORK_DIR / "video_concat.txt"
    build_concat_file(all_images, video_list, all_durations)
    video_silent = OUTPUT_DIR / "video_silent.mp4"

    motion_x = 42 if MOTION_INTENSITY == "1.0" else 30
    motion_y = 28 if MOTION_INTENSITY == "1.0" else 20
    emphasis_filter = _make_emphasis_filter(scene_audio_durations)
    vf_parts = [
        "scale=1500:844:force_original_aspect_ratio=increase",
        f"crop={VIDEO_W}:{VIDEO_H}:x='58+{motion_x}*sin(2*PI*t/8.5)+14*sin(2*PI*t/2.8)':y='32+{motion_y}*cos(2*PI*t/10.5)+8*sin(2*PI*t/3.6)'",
        "eq=contrast=1.03:saturation=1.06",
        "unsharp=5:5:0.8:5:5:0.35",
        "noise=alls=2:allf=t+u",
    ]
    if emphasis_filter:
        vf_parts.append(emphasis_filter)
    vf_parts.append("format=yuv420p")

    run_cmd(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(video_list),
            "-vf", ",".join(vf_parts),
            "-r", str(VIDEO_FPS), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-pix_fmt", "yuv420p", str(video_silent),
        ],
        "render motion-comic sequence with reveal graphics",
    )

    audio_list = WORK_DIR / "audio_concat.txt"
    build_concat_file(audio_paths, audio_list)
    full_audio = OUTPUT_DIR / "full_audio.wav"
    run_cmd(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(audio_list),
            "-ar", str(AUDIO_SR), "-ac", "1", "-c:a", "pcm_s16le", str(full_audio),
        ],
        "concatenate narration",
    )

    mixed_audio = OUTPUT_DIR / "full_audio_mixed.m4a"
    sound_design = _make_sound_design_track(total, scene_audio_durations, all_durations)

    if MUSIC_PATH.exists():
        inputs = ["-i", str(full_audio), "-stream_loop", "-1", "-i", str(MUSIC_PATH)]
        if sound_design:
            inputs += ["-i", str(sound_design)]
        if sound_design:
            filter_complex = (
                "[0:a]highpass=f=70,acompressor=threshold=-18dB:ratio=2.6:attack=5:release=120:makeup=2,loudnorm=I=-15.5:TP=-1.5:LRA=8[n];"
                f"[1:a]volume={BACKGROUND_MUSIC_VOLUME:.3f},highpass=f=90,lowpass=f=9000[m];"
                "[m][n]sidechaincompress=threshold=0.03:ratio=6:attack=25:release=450:makeup=1[ducked];"
                "[n][ducked]amix=inputs=2:duration=first:dropout_transition=2[nm];"
                "[2:a]volume=0.9[sfx];"
                "[nm][sfx]amix=inputs=2:duration=first:dropout_transition=1[a]"
            )
        else:
            filter_complex = (
                "[0:a]highpass=f=70,acompressor=threshold=-18dB:ratio=2.6:attack=5:release=120:makeup=2,loudnorm=I=-15.5:TP=-1.5:LRA=8[n];"
                f"[1:a]volume={BACKGROUND_MUSIC_VOLUME:.3f},highpass=f=90,lowpass=f=9000[m];"
                "[m][n]sidechaincompress=threshold=0.03:ratio=6:attack=25:release=450:makeup=1[ducked];"
                "[n][ducked]amix=inputs=2:duration=first:dropout_transition=2[a]"
            )
        run_cmd(
            ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-map", "[a]",
             "-c:a", "aac", "-b:a", "192k", str(mixed_audio)],
            "mix narration, music, and sound design",
        )
    else:
        if sound_design:
            run_cmd(
                [
                    "ffmpeg", "-y", "-i", str(full_audio), "-i", str(sound_design),
                    "-filter_complex",
                    "[0:a]highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11[n];[1:a]volume=0.9[sfx];[n][sfx]amix=inputs=2:duration=first:dropout_transition=1[a]",
                    "-map", "[a]", "-c:a", "aac", "-b:a", "192k", str(mixed_audio),
                ],
                "normalize narration with sound design",
            )
        else:
            run_cmd(
                [
                    "ffmpeg", "-y", "-i", str(full_audio),
                    "-af", "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-c:a", "aac", "-b:a", "192k", str(mixed_audio),
                ],
                "normalize narration",
            )

    run_cmd(
        [
            "ffmpeg", "-y", "-i", str(video_silent), "-i", str(mixed_audio),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path),
        ],
        "mux final video",
    )
    print(f"[VIDEO] ready: {out_path} (~{total / 60:.1f} min, {len(all_images)} visual beats, premium motion/audio enabled)")
    checkpoint("video_complete", duration_seconds=round(total, 2), visual_beats=len(all_images),
               sound_design=bool(sound_design), emphasis_cards=ENABLE_EMPHASIS_CARDS)
    return total


# ---------------------------------------------------------------------------
# YouTube upload
# ---------------------------------------------------------------------------
def upload_video(video_path: Path, thumbnail_path: Path, seo: dict[str, Any]) -> str:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    token_text = require_secret("YOUTUBE_TOKEN_JSON")
    client_secret_text = require_secret("YOUTUBE_CLIENT_SECRET_JSON")
    token_path = ROOT / "youtube_token.json"
    client_secret_path = ROOT / "client_secret.json"
    token_path.write_text(token_text, encoding="utf-8")
    client_secret_path.write_text(client_secret_text, encoding="utf-8")

    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    credentials = Credentials.from_authorized_user_file(str(token_path), scopes)
    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            raise RuntimeError("YouTube token is invalid/expired and has no usable refresh token.")

    youtube = build("youtube", "v3", credentials=credentials)

    # Avoid duplicate uploads if a previous runner uploaded before another stage failed.
    existing_upload = _load_current_json(CURRENT_UPLOAD_PATH)
    if isinstance(existing_upload, dict) and existing_upload.get("video_id"):
        video_id = str(existing_upload["video_id"])
        print(f"[YOUTUBE] Reusing previously uploaded video {video_id}")
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path),
            ).execute()
            print("[YOUTUBE] thumbnail set on reused upload")
        except Exception as exc:
            print(f"[YOUTUBE] thumbnail upload warning: {exc}")
        return video_id

    body = {
        "snippet": {
            "title": seo["title"][:100],
            "description": seo["description"][:4900],
            "tags": seo["tags"][:15],
            "categoryId": "24",
        },
        "status": {
            "privacyStatus": YOUTUBE_PRIVACY_STATUS,
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=-1, resumable=True)
    response = youtube.videos().insert(part="snippet,status", body=body, media_body=media).execute()
    video_id = response["id"]
    _save_current_json(CURRENT_UPLOAD_PATH, {
        "video_id": video_id,
        "privacy_status": YOUTUBE_PRIVACY_STATUS,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"[YOUTUBE] uploaded {video_id} ({YOUTUBE_PRIVACY_STATUS})")

    try:
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg"),
        ).execute()
        print("[YOUTUBE] thumbnail set")
    except Exception as exc:
        print(f"[YOUTUBE] thumbnail upload warning: {exc}")

    return video_id


def validate_youtube_credentials() -> None:
    """Preflight the stored YouTube OAuth credentials before expensive work."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    token_text = require_secret("YOUTUBE_TOKEN_JSON")
    client_secret_text = require_secret("YOUTUBE_CLIENT_SECRET_JSON")

    try:
        token_info = json.loads(token_text)
        client_info = json.loads(client_secret_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("YouTube credential secrets must contain valid JSON.") from exc

    if not isinstance(token_info, dict):
        raise RuntimeError("YOUTUBE_TOKEN_JSON must contain a JSON object.")
    if not isinstance(client_info, dict):
        raise RuntimeError("YOUTUBE_CLIENT_SECRET_JSON must contain a JSON object.")

    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    credentials = Credentials.from_authorized_user_info(token_info, scopes)

    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except Exception as exc:
                raise RuntimeError(
                    "YouTube OAuth refresh failed. The saved refresh token may be expired or revoked."
                ) from exc
        else:
            raise RuntimeError(
                "YouTube OAuth credentials are invalid and do not have a usable refresh token."
            )

    print("[YOUTUBE PREFLIGHT] OAuth credentials are usable.")



# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------
def update_history(topic: dict[str, Any], seo: dict[str, Any], video_id: str | None) -> None:
    history = load_history()
    item = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "question": topic.get("question"),
        "topic": topic.get("topic"),
        "era": topic.get("era"),
        "title": seo.get("title"),
        "video_id": video_id,
    }
    history["videos"].append(item)
    # Keep the file useful but bounded.
    history["videos"] = history["videos"][-200:]
    save_history(history)


def save_manifests(topic: dict[str, Any], research: str, story: dict[str, Any], script: dict[str, Any], seo: dict[str, Any]) -> None:
    atomic_write_json(TOPIC_PATH, topic)
    RESEARCH_PATH.write_text(research, encoding="utf-8")
    atomic_write_json(STORY_PATH, story)
    atomic_write_json(SCRIPT_PATH, script)
    atomic_write_json(SEO_PATH, seo)
    atomic_write_json(
        MANIFEST_PATH,
        {
            "topic": topic,
            "script_file": str(SCRIPT_PATH.relative_to(ROOT)),
            "scene_count": len(script.get("scenes", [])),
            "title": seo.get("title"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def main(mode: str = "full") -> None:
    print(f"=== {CHANNEL_NAME} / Relic Loop v6 ===")
    print(f"MODE={mode} | KOKORO_VOICE={KOKORO_VOICE} | SPEED={KOKORO_SPEED}")

    if mode == "voice_test":
        voice_test()
        return

    if mode == "visual_test":
        visual_test()
        return

    if mode == "local_image_test":
        local_image_test()
        return

    require_secret("GROQ_API_KEY")
    require_secret("YOUTUBE_TOKEN_JSON")
    require_secret("YOUTUBE_CLIENT_SECRET_JSON")
    supported = {"cloudflare", "huggingface", "replicate", "local"}
    unknown = [x for x in IMAGE_PROVIDER_ORDER if x not in supported]
    if unknown:
        raise RuntimeError(f"Unsupported image providers: {unknown}")
    if "cloudflare" in IMAGE_PROVIDER_ORDER:
        require_secret("CLOUDFLARE_ACCOUNT_ID")
        require_secret("CLOUDFLARE_API_TOKEN")

    validate_youtube_credentials()

    checkpoint("starting")
    history = load_history()

    # Reuse small text artifacts saved by a previous failed Actions run.
    topic = _load_current_json(CURRENT_TOPIC_PATH)
    topic_resumed = isinstance(topic, dict) and bool(topic.get("question"))
    if topic_resumed:
        print("[RESUME] Reusing saved topic.")
    else:
        topic = choose_topic(history)
        _save_current_json(CURRENT_TOPIC_PATH, topic)
    checkpoint("topic_complete", question=topic["question"], resumed=topic_resumed)

    research = _load_current_text(CURRENT_RESEARCH_PATH)
    research_resumed = bool(research)
    if research_resumed:
        print("[RESUME] Reusing saved research.")
    else:
        research = research_topic(topic)
        _save_current_text(CURRENT_RESEARCH_PATH, research)
    RESEARCH_PATH.write_text(research, encoding="utf-8")
    checkpoint("research_complete", resumed=research_resumed)

    story = _load_current_json(CURRENT_STORY_PATH)
    story_resumed = isinstance(story, dict) and bool(story.get("story_arc"))
    if story_resumed:
        print("[RESUME] Reusing saved story plan.")
    else:
        story = build_story_plan(topic, research)
        _save_current_json(CURRENT_STORY_PATH, story)
    atomic_write_json(STORY_PATH, story)
    checkpoint("story_plan_complete", resumed=story_resumed)

    script = _load_current_json(CURRENT_SCRIPT_PATH)
    if isinstance(script, dict):
        try:
            validate_script(script, allow_short=True)
            saved_words = _script_word_count(script)
            if SCRIPT_MIN_WORDS <= saved_words <= SCRIPT_MAX_WORDS:
                print(f"[RESUME] Reusing saved validated script (~{saved_words} words).")
            else:
                print(f"[RESUME] Reusing saved partial script (~{saved_words} words) and continuing repair.")
                script = _repair_script_length(topic, research, story, script)
                _save_current_json(CURRENT_SCRIPT_PATH, script)
        except Exception as exc:
            print(f"[RESUME] Saved script invalid; regenerating: {exc}")
            script = None

    if script is None:
        script = write_script(topic, research, story)


    atomic_write_json(SCRIPT_PATH, script)
    checkpoint("script_complete", scene_count=len(script["scenes"]), words=_script_word_count(script))

    # Carry topic metadata into the visual director so modern/science/everyday
    # episodes do not inherit historical defaults.
    script["topic_category"] = str(topic.get("category", "general"))
    script["central_question"] = str(topic.get("question", ""))

    # Audio/scenes are regenerated on fresh runners from the saved script.
    generate_voiceovers(script)
    render_scenes(script)

    seo = _load_current_json(CURRENT_SEO_PATH)
    seo_resumed = isinstance(seo, dict) and bool(seo.get("title") and seo.get("description") and seo.get("tags"))
    if seo_resumed:
        print("[RESUME] Reusing saved SEO package.")
    else:
        seo = build_seo(topic, script, research)
        _save_current_json(CURRENT_SEO_PATH, seo)
    atomic_write_json(SEO_PATH, seo)
    checkpoint("seo_complete", title=seo["title"], resumed=seo_resumed)

    package_issues = package_quality_issues(topic, script, seo)
    if package_issues:
        print(f"[PACKAGE] quality warnings: {package_issues}")
        checkpoint("package_quality_warning", issues=package_issues)

    thumb = make_thumbnail(script, seo["title"], seo.get("thumbnail_headline"))
    video_path = OUTPUT_DIR / "final_video.mp4"
    build_video(script, video_path)

    video_id = upload_video(video_path, thumb, seo)
    update_history(topic, seo, video_id)
    atomic_write_json(
        MANIFEST_PATH,
        {
            "topic": topic,
            "title": seo["title"],
            "video_id": video_id,
            "privacy_status": YOUTUBE_PRIVACY_STATUS,
            "scene_count": len(script["scenes"]),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    clear_current_run()
    checkpoint("complete", video_id=video_id, title=seo["title"])
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "voice_test", "visual_test", "local_image_test"],
        default="full",
    )
    args = parser.parse_args()
    main(args.mode)
