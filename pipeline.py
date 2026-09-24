"""
Motive Unknown v2 — curiosity-first automated history video factory.

Design goals
------------
- Daily, unattended GitHub Actions execution.
- Curiosity-driven historical questions instead of generic topics.
- GPT-OSS 120B for research/storytelling; GPT-OSS 20B for lightweight
  structuring/SEO tasks.
- Local/open-weight Kokoro TTS (no paid voice API).
- No per-clip image API calls. A deterministic stickman/storybook renderer
  creates illustrated scenes locally, eliminating image-provider 429 loops.
- No Ken Burns zoom and no burned-in subtitles.
- Dedicated thumbnail generation separate from video scenes.
- Idempotent stage files so a rerun can skip already-completed stages.
- Safe YouTube default: private uploads until the owner changes the setting.

Required GitHub Secrets
-----------------------
GROQ_API_KEY
YOUTUBE_TOKEN_JSON
YOUTUBE_CLIENT_SECRET_JSON

Optional GitHub Variables / Secrets
-----------------------------------
YOUTUBE_PRIVACY_STATUS     default: private
KOKORO_VOICE               default: am_onyx
KOKORO_SPEED               default: 0.96
CHANNEL_NAME               optional, used in prompts/description

Optional repo asset
-------------------
assets/music/background.mp3  royalty-free / licensed music only

The workflow file supplied with this package runs daily and also supports a
manual "voice_test" mode before committing to a full production run.
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
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
WORK_DIR = ROOT / "run_work"
SCENE_DIR = WORK_DIR / "scenes"
AUDIO_DIR = WORK_DIR / "audio"
THUMB_DIR = WORK_DIR / "thumbnails"
OUTPUT_DIR = WORK_DIR / "output"

for d in (STATE_DIR, WORK_DIR, SCENE_DIR, AUDIO_DIR, THUMB_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_RESEARCH_MODEL = os.environ.get("GROQ_RESEARCH_MODEL", "openai/gpt-oss-120b")
GROQ_WRITER_MODEL = os.environ.get("GROQ_WRITER_MODEL", "openai/gpt-oss-120b")
GROQ_LIGHT_MODEL = os.environ.get("GROQ_LIGHT_MODEL", "openai/gpt-oss-20b")

KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "am_onyx").strip()
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", "0.96"))

CHANNEL_NAME = os.environ.get("CHANNEL_NAME", "Motive Unknown").strip()
YOUTUBE_PRIVACY_STATUS = os.environ.get("YOUTUBE_PRIVACY_STATUS", "private").strip().lower()
if YOUTUBE_PRIVACY_STATUS not in {"private", "public", "unlisted"}:
    YOUTUBE_PRIVACY_STATUS = "private"

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
    # Do not crash voice_test if someone simply imports this file in an IDE.
    client: Groq | None = None
else:
    client = Groq(api_key=GROQ_API_KEY)


def groq_call(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_completion_tokens: int,
    temperature: float = 0.6,
    browser_search: bool = False,
    attempts: int = 5,
) -> str:
    if client is None:
        raise RuntimeError("GROQ_API_KEY is required for this step.")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens,
                "temperature": temperature,
                "include_reasoning": False,
            }
            if browser_search:
                kwargs["tools"] = [{"type": "browser_search"}]
                kwargs["tool_choice"] = "required"
                kwargs["citation_options"] = "disabled"

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise RuntimeError("Groq returned an empty response.")
            return content.strip()
        except Exception as exc:  # SDK raises distinct subclasses depending on failure
            last_error = exc
            message = str(exc)
            # Groq's SDK may surface 429 as a plain exception string.
            wait = 10.0 * attempt
            match = re.search(r"try again in ([\d.]+)s", message, flags=re.IGNORECASE)
            if match:
                wait = max(float(match.group(1)) + 1.0, wait)
            print(f"[GROQ RETRY] {model} attempt {attempt}/{attempts}: {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Groq call failed after {attempts} attempts: {last_error}")


def groq_json(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_completion_tokens: int,
    temperature: float,
    attempts: int = 3,
) -> dict[str, Any]:
    if client is None:
        raise RuntimeError("GROQ_API_KEY is required for this step.")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                include_reasoning=False,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            return extract_last_json_object(clean_json_text(raw))
        except Exception as exc:
            last_error = exc
            wait = 8 * attempt
            match = re.search(r"try again in ([\d.]+)s", str(exc), flags=re.IGNORECASE)
            if match:
                wait = max(wait, float(match.group(1)) + 1)
            print(f"[GROQ JSON RETRY] {model} attempt {attempt}/{attempts}: {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Groq JSON call failed: {last_error}")


# ---------------------------------------------------------------------------
# Topic scouting / research
# ---------------------------------------------------------------------------
TOPIC_PROMPT = """
You are a curiosity-first history channel producer.

Your job is NOT to pick a generic history topic. Generate one specific historical
QUESTION that a normal person might genuinely wonder about after hearing it.

Good patterns:
- Why did ...?
- How did ...?
- What happened when ...?
- Why were ... so important to ...?
- How did an ordinary decision cause ...?
- Why did people suddenly start/stop ...?
- What was really happening behind ...?

Prefer stories involving a famous civilization, ruler, event, object, belief, or
place when there is a genuinely strange or surprising question attached to it.
Avoid fake mysteries, conspiracy framing, paranormal claims presented as fact,
and "history of X" topics.

The eventual video should be understandable to an intelligent young teenager,
while still being interesting to adults.

Return exactly this JSON object:
{
  "question": "one irresistible historical question",
  "topic": "short internal topic label",
  "era": "short era/civilization label",
  "why_curious": "2-4 sentences explaining why the question creates curiosity",
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

    raw = groq_call(
        GROQ_LIGHT_MODEL,
        [
            {
                "role": "user",
                "content": TOPIC_PROMPT
                + "\n\nPreviously used topics/questions. Avoid these and close variations:\n"
                + history_text,
            }
        ],
        max_completion_tokens=1100,
        temperature=0.8,
        browser_search=True,
    )
    data = extract_last_json_object(clean_json_text(raw))
    for key in ("question", "topic", "era", "why_curious", "search_angles"):
        if not data.get(key):
            raise RuntimeError(f"Topic scout missing field: {key}")
    if count_words(data["question"]) < 4:
        raise RuntimeError("Topic question is too short.")
    return data


def research_topic(topic: dict[str, Any]) -> str:
    prompt = f"""
You are the lead historical researcher for a documentary channel.

Central question:
{topic['question']}

Topic:
{topic['topic']}

Era/civilization:
{topic['era']}

Research angles:
{json.dumps(topic.get('search_angles', []), ensure_ascii=False)}

Use browser search extensively. Prefer reliable sources such as museums,
universities, national archives, reputable reference works, academic/history
institutions, and well-maintained reference pages. Wikipedia is acceptable as a
starting point but should not be the only authority for a surprising claim.

Return a compact research dossier in plain text with these headings:
1. CORE ANSWER
2. STORY BEATS (10-16 numbered beats)
3. IMPORTANT PEOPLE / PLACES / OBJECTS
4. DISPUTES OR UNCERTAINTY
5. SOURCES (at least 6 sources with title + URL)

Do not optimize for shock. Optimize for a fascinating question that can be
answered honestly. Clearly flag claims that are uncertain, legendary, or disputed.
The final script will be factual and non-graphic.
""".strip()
    return groq_call(
        GROQ_RESEARCH_MODEL,
        [{"role": "user", "content": prompt}],
        max_completion_tokens=4200,
        temperature=0.45,
        browser_search=True,
        attempts=5,
    )


# ---------------------------------------------------------------------------
# Story architecture + script
# ---------------------------------------------------------------------------
STORY_ARCHITECT_PROMPT = """
You are a story architect for a premium history YouTube channel.

Turn the research dossier into a suspenseful STORY PLAN, not a textbook outline.
The video must have one central curiosity question and a satisfying answer.

Structure:
1. Cold open: start with the strangest or most important moment, without a long intro.
2. Immediate question: make the viewer understand what they are trying to figure out.
3. Minimal context: only the background needed to understand the mystery.
4. Escalation: each section should reveal something that changes the viewer's picture.
5. Midpoint turn: a fact, decision, discovery, or contradiction that surprises us.
6. Deepening: show why the obvious explanation is not enough.
7. Payoff: answer the central question using the strongest evidence.
8. Ending: leave the viewer with one memorable implication or final twist grounded in fact.

Avoid chronological padding. Avoid atmosphere-for-atmosphere's-sake.

Return JSON:
{
  "central_question": "...",
  "hook": "1-3 sentence cold open concept",
  "story_arc": [
    {"section": 1, "purpose": "...", "reveal": "...", "required_facts": ["..."]}
  ],
  "ending_payoff": "..."
}
""".strip()


def build_story_plan(topic: dict[str, Any], research: str) -> dict[str, Any]:
    prompt = (
        STORY_ARCHITECT_PROMPT
        + "\n\nTOPIC:\n"
        + json.dumps(topic, ensure_ascii=False)
        + "\n\nRESEARCH DOSSIER:\n"
        + research
    )
    plan = groq_json(
        GROQ_LIGHT_MODEL,
        [{"role": "user", "content": prompt}],
        max_completion_tokens=3500,
        temperature=0.55,
    )
    if not plan.get("story_arc") or len(plan["story_arc"]) < 8:
        raise RuntimeError("Story architect returned too little structure.")
    return plan


SCRIPTWRITER_PROMPT = """
You are the head writer of an excellent history storytelling channel.

Write a 10-15 minute narration that answers one irresistible historical question.
The target audience is a curious young teenager AND adults. The language is simple,
but the thinking is not childish.

VOICE OF THE WRITING
- Modern, conversational, confident, vivid.
- Sounds like a brilliant storyteller talking directly to the viewer.
- Short and medium sentences mixed for rhythm.
- Use concrete actions and decisions instead of decorative prose.
- Reveal information in the order that creates curiosity.
- Ask a question only when it genuinely advances the story.
- Use humor lightly when it fits the facts.
- Let characters make choices and let consequences matter.
- Make the viewer feel like they are discovering the answer with you.

DO NOT WRITE LIKE
- a school essay
- an old-fashioned novel
- an encyclopedia
- a travel brochure
- a movie trailer full of fake suspense

Avoid filler such as "the sun was shining," "the water was calm," "little did they know,"
"in the annals of history," and long scenery descriptions unless the detail changes the story.
Never add a fact simply to make the script longer.
Never invent dialogue or inner thoughts and present them as historical facts.
When evidence is uncertain or disputed, say so naturally.
Do not use graphic descriptions.

VERY IMPORTANT: the first 20-30 seconds must make a viewer who has never heard of this
story think: "Wait, why did that happen?"

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
    "headline": "2-5 words, not the full title",
    "subject": "main visual subject",
    "supporting_prop": "one strong prop or symbol",
    "emotion": "clear facial/body emotion",
    "composition": "left_subject_right_prop / right_subject_left_prop / central_subject",
    "preferred_variant": 1
  }
}

Scene count: 18-38.
Total narration: 1700-2600 words.
Each scene must describe a distinct, useful visual moment. Do not create a new scene just
because a sentence changed. Scenes can hold for several seconds.
""".strip()


def write_script(topic: dict[str, Any], research: str, plan: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        SCRIPTWRITER_PROMPT
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
        max_completion_tokens=7000,
        temperature=0.78,
        attempts=4,
    )
    validate_script(script)
    return script


def validate_script(script: dict[str, Any]) -> None:
    scenes = script.get("scenes")
    if not isinstance(scenes, list) or not (SCENE_MIN <= len(scenes) <= SCENE_MAX):
        raise RuntimeError(f"Script scene count must be {SCENE_MIN}-{SCENE_MAX}; got {len(scenes) if isinstance(scenes, list) else 0}.")
    total_words = 0
    for i, scene in enumerate(scenes, 1):
        if not scene.get("narration"):
            raise RuntimeError(f"Scene {i} is missing narration.")
        words = count_words(scene["narration"])
        if words < 25 or words > 120:
            raise RuntimeError(f"Scene {i} narration is {words} words; expected 25-120.")
        for key in ("setting", "characters", "action", "props", "mood"):
            if key not in scene:
                raise RuntimeError(f"Scene {i} missing {key}.")
        total_words += words
    if not (SCRIPT_MIN_WORDS <= total_words <= SCRIPT_MAX_WORDS):
        raise RuntimeError(f"Total script word count {total_words} outside {SCRIPT_MIN_WORDS}-{SCRIPT_MAX_WORDS}.")
    print(f"[SCRIPT] validated: {len(scenes)} scenes, ~{total_words} words")


# ---------------------------------------------------------------------------
# Kokoro TTS
# ---------------------------------------------------------------------------
_kokoro_pipeline = None


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


def synthesize_kokoro(text: str) -> np.ndarray:
    pipeline = get_kokoro_pipeline()
    chunks: list[np.ndarray] = []
    generator = pipeline(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED)
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


def voice_test() -> Path:
    sample = (
        "For years, historians thought they knew the answer. Then one tiny detail "
        "started causing problems. The people involved had left clues behind. The clues "
        "looked ordinary. But put them together, and the story suddenly changes. Why? "
        "Because the obvious explanation leaves one very strange question unanswered."
    )
    out = ROOT / "voice_test_onyx.wav"
    audio = synthesize_kokoro(sample)
    sf.write(out, audio, AUDIO_SR)
    print(f"[VOICE TEST] wrote {out} ({len(audio) / AUDIO_SR:.1f}s) using {KOKORO_VOICE}")
    return out


# ---------------------------------------------------------------------------
# Stickman / storybook renderer
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


def era_theme(era: str, setting: str) -> dict[str, Any]:
    all_text = f"{era} {setting}".lower()
    if contains_any(all_text, ["egypt", "pharaoh", "nile", "pyramid"]):
        return {"sky": (244, 226, 181), "ground": SAND, "accent": GOLD, "building": (194, 165, 105), "water": (98, 160, 180), "kind": "egypt"}
    if contains_any(all_text, ["china", "chinese", "han", "qin", "ming", "imperial"]):
        return {"sky": (219, 231, 244), "ground": (193, 181, 151), "accent": RED, "building": (174, 88, 69), "water": (93, 146, 178), "kind": "china"}
    if contains_any(all_text, ["rome", "roman", "latin"]):
        return {"sky": (222, 229, 235), "ground": (178, 168, 149), "accent": RED, "building": (170, 157, 139), "water": (100, 145, 167), "kind": "rome"}
    if contains_any(all_text, ["viking", "norse", "scandinavia"]):
        return {"sky": (193, 216, 229), "ground": (146, 164, 145), "accent": BLUE, "building": (130, 106, 82), "water": (82, 133, 166), "kind": "viking"}
    if contains_any(all_text, ["japan", "japanese", "shogun", "samurai"]):
        return {"sky": (231, 221, 221), "ground": (154, 181, 135), "accent": RED, "building": (152, 95, 78), "water": (95, 146, 177), "kind": "japan"}
    if contains_any(all_text, ["aztec", "maya", "inca", "mesoamerica"]):
        return {"sky": (212, 229, 214), "ground": (131, 174, 111), "accent": GOLD, "building": (154, 131, 91), "water": (92, 155, 170), "kind": "meso"}
    if contains_any(all_text, ["medieval", "castle", "europe", "kingdom"]):
        return {"sky": (216, 226, 236), "ground": (163, 184, 144), "accent": PURPLE, "building": (136, 141, 151), "water": (96, 148, 174), "kind": "medieval"}
    return {"sky": SKY, "ground": GRASS, "accent": BLUE, "building": STONE, "water": (100, 152, 174), "kind": "generic"}


def draw_background(draw: ImageDraw.ImageDraw, theme: dict[str, Any], setting: str) -> None:
    draw.rectangle([0, 0, VIDEO_W, VIDEO_H], fill=theme["sky"])
    draw.rectangle([0, 470, VIDEO_W, VIDEO_H], fill=theme["ground"])
    # Sun/moon
    draw.ellipse([1070, 70, 1170, 170], fill=(245, 208, 102))

    s = setting.lower()
    kind = theme["kind"]
    if contains_any(s, ["river", "nile", "harbor", "sea", "ship", "lake"]):
        draw.rectangle([0, 385, VIDEO_W, 500], fill=theme["water"])
    elif contains_any(s, ["desert", "sand"]):
        draw.rectangle([0, 405, VIDEO_W, VIDEO_H], fill=SAND)
    elif contains_any(s, ["palace", "temple", "court", "throne"]):
        draw.rectangle([80, 180, 1200, 475], fill=theme["building"])
        draw.rectangle([130, 260, 240, 470], fill=(145, 131, 112))
        draw.rectangle([1040, 260, 1150, 470], fill=(145, 131, 112))
        draw.polygon([(540, 180), (640, 90), (740, 180)], fill=theme["accent"])
        draw.rectangle([570, 285, 710, 470], fill=(108, 87, 68))
    elif contains_any(s, ["street", "market", "town", "city"]):
        for x in [100, 350, 720, 1020]:
            draw.polygon([(x, 270), (x + 90, 210), (x + 180, 270)], fill=theme["building"])
            draw.rectangle([x + 20, 270, x + 160, 470], fill=theme["building"])
            draw.rectangle([x + 72, 350, x + 110, 470], fill=theme["sky"])
    elif contains_any(s, ["battlefield", "field", "camp"]):
        for x in range(80, 1250, 160):
            draw.line([x, 470, x + 40, 330], fill=BROWN, width=8)
            draw.polygon([(x + 40, 330), (x + 90, 355), (x + 40, 380)], fill=theme["accent"])
    elif kind == "egypt":
        draw.polygon([(130, 470), (290, 270), (450, 470)], fill=(201, 173, 112))
        draw.polygon([(760, 470), (900, 300), (1040, 470)], fill=(204, 176, 116))
    elif kind == "china":
        draw.rectangle([120, 260, 1160, 470], fill=theme["building"])
        for x in range(170, 1130, 120):
            draw.polygon([(x, 250), (x + 40, 210), (x + 80, 250)], fill=RED)
            draw.rectangle([x + 30, 250, x + 50, 470], fill=(115, 79, 55))
    elif kind == "rome":
        for x in [140, 300, 460, 800, 960, 1120]:
            draw.rectangle([x, 220, x + 55, 470], fill=theme["building"])
            draw.arc([x - 10, 175, x + 65, 260], 180, 360, fill=theme["building"], width=10)
    elif kind == "japan":
        draw.polygon([(200, 290), (360, 210), (520, 290)], fill=theme["building"])
        draw.rectangle([260, 290, 460, 470], fill=theme["building"])
        draw.ellipse([830, 190, 1050, 410], outline=RED, width=24)
    else:
        # Light decorative clouds
        for x in [140, 490, 910]:
            draw.ellipse([x, 120, x + 90, 170], fill=(255, 255, 255))
            draw.ellipse([x + 40, 105, x + 120, 165], fill=(255, 255, 255))

    # Storybook border
    draw.rectangle([16, 16, VIDEO_W - 16, VIDEO_H - 16], outline=INK, width=5)


def archetype_style(label: str, era: str) -> dict[str, Any]:
    t = f"{label} {era}".lower()
    if contains_any(t, ["pharaoh", "egyptian", "scribe"]):
        return {"body": (228, 198, 150), "outfit": GOLD if "pharaoh" in t else WHITE, "hat": GOLD if "pharaoh" in t else None, "kind": "egypt"}
    if contains_any(t, ["roman", "legionary", "centurion"]):
        return {"body": (222, 184, 141), "outfit": RED, "hat": STONE, "kind": "rome"}
    if contains_any(t, ["chinese", "emperor", "courtier", "mandarin"]):
        return {"body": (228, 192, 155), "outfit": RED if "emperor" in t else BLUE, "hat": BLACK, "kind": "china"}
    if contains_any(t, ["viking", "norse", "warrior"]):
        return {"body": (216, 179, 143), "outfit": BROWN, "hat": STONE, "kind": "viking"}
    if contains_any(t, ["samurai", "japanese", "shogun"]):
        return {"body": (218, 180, 148), "outfit": BLACK if "samurai" in t else RED, "hat": BLACK, "kind": "japan"}
    if contains_any(t, ["king", "queen", "monarch", "noble"]):
        return {"body": (222, 181, 146), "outfit": PURPLE, "hat": GOLD, "kind": "royal"}
    return {"body": (224, 185, 148), "outfit": BLUE, "hat": None, "kind": "generic"}


def draw_stickman(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    scale: float,
    label: str,
    era: str,
    action: str,
    flip: bool = False,
) -> None:
    style = archetype_style(label, era)
    r = int(24 * scale)
    body = style["body"]
    outfit = style["outfit"]
    head_y = y - int(140 * scale)
    torso_y = y - int(85 * scale)
    hip_y = y - int(25 * scale)

    # Head
    draw.ellipse([x - r, head_y - r, x + r, head_y + r], fill=body, outline=INK, width=max(2, int(4 * scale)))

    # Hair / hat
    if style["kind"] in {"china", "viking", "japan", "royal"} and style["hat"]:
        draw.rectangle([x - r - 5, head_y - r - 8, x + r + 5, head_y - r + 6], fill=style["hat"], outline=INK, width=2)
    if style["kind"] == "egypt":
        draw.polygon([(x - r - 8, head_y - r), (x, head_y - r - 30), (x + r + 8, head_y - r)], fill=style["hat"] or GOLD, outline=INK)

    # Face
    draw.ellipse([x - int(8 * scale), head_y - 5, x - int(4 * scale), head_y - 1], fill=INK)
    draw.ellipse([x + int(4 * scale), head_y - 5, x + int(8 * scale), head_y - 1], fill=INK)

    # Torso / costume
    draw.line([x, head_y + r, x, hip_y], fill=INK, width=max(4, int(6 * scale)))
    draw.line([x - int(18 * scale), torso_y, x + int(18 * scale), torso_y], fill=outfit, width=max(8, int(15 * scale)))
    # Cape / robe for royal or emperor-like characters.
    if style["kind"] == "royal" or contains_any(label, ["emperor", "pharaoh", "shogun"]):
        draw.polygon(
            [(x - int(18 * scale), torso_y), (x - int(45 * scale), hip_y), (x + int(45 * scale), hip_y), (x + int(18 * scale), torso_y)],
            fill=outfit,
            outline=INK,
        )

    act = action.lower()
    raise_arm = contains_any(act, ["raise", "hold up", "lift", "signal", "point", "pointing"])
    hold_item = contains_any(act, ["hold", "carry", "read", "show", "present"])
    wave = contains_any(act, ["wave", "greet"])

    if flip:
        direction = -1
    else:
        direction = 1

    arm_dx = int(55 * scale)
    if raise_arm:
        draw.line([x, torso_y, x + direction * arm_dx, torso_y - int(60 * scale)], fill=INK, width=max(4, int(5 * scale)))
        draw.line([x, torso_y, x - direction * int(40 * scale), torso_y + int(25 * scale)], fill=INK, width=max(4, int(5 * scale)))
    elif wave:
        draw.line([x, torso_y, x + direction * arm_dx, torso_y - int(25 * scale)], fill=INK, width=max(4, int(5 * scale)))
        draw.line([x + direction * arm_dx, torso_y - int(25 * scale), x + direction * int(75 * scale), torso_y - int(80 * scale)], fill=INK, width=max(4, int(5 * scale)))
    else:
        draw.line([x, torso_y, x - direction * int(42 * scale), torso_y + int(30 * scale)], fill=INK, width=max(4, int(5 * scale)))
        draw.line([x, torso_y, x + direction * int(42 * scale), torso_y + int(30 * scale)], fill=INK, width=max(4, int(5 * scale)))

    # Legs
    walking = contains_any(act, ["walk", "run", "leave", "move", "approach"])
    if walking:
        draw.line([x, hip_y, x - int(38 * scale), hip_y + int(70 * scale)], fill=INK, width=max(4, int(6 * scale)))
        draw.line([x, hip_y, x + int(50 * scale), hip_y + int(55 * scale)], fill=INK, width=max(4, int(6 * scale)))
    else:
        draw.line([x, hip_y, x - int(30 * scale), hip_y + int(75 * scale)], fill=INK, width=max(4, int(6 * scale)))
        draw.line([x, hip_y, x + int(30 * scale), hip_y + int(75 * scale)], fill=INK, width=max(4, int(6 * scale)))

    # Tiny accessory cues make archetypes readable.
    if contains_any(label, ["soldier", "warrior", "guard", "samurai", "roman"]):
        draw.line([x + direction * int(50 * scale), torso_y + int(10 * scale), x + direction * int(65 * scale), torso_y - int(55 * scale)], fill=INK, width=max(2, int(4 * scale)))
    if contains_any(label, ["scribe", "scholar", "monk", "student"]):
        draw.rectangle([x + direction * int(28 * scale), torso_y + int(22 * scale), x + direction * int(60 * scale), torso_y + int(45 * scale)], fill=PAPER, outline=INK)


def draw_prop(draw: ImageDraw.ImageDraw, x: int, y: int, prop: str, scale: float = 1.0) -> None:
    p = prop.lower()
    if contains_any(p, ["scroll", "document", "letter", "papyrus"]):
        draw.rectangle([x - 35, y - 12, x + 35, y + 12], fill=PAPER, outline=INK, width=3)
        draw.arc([x - 45, y - 25, x - 15, y + 5], 90, 270, fill=BROWN, width=4)
        draw.arc([x + 15, y - 5, x + 45, y + 25], 270, 90, fill=BROWN, width=4)
    elif contains_any(p, ["sword", "blade"]):
        draw.line([x - 10, y + 40, x + 55, y - 40], fill=STONE, width=8)
        draw.line([x - 5, y + 20, x + 18, y + 43], fill=BROWN, width=8)
    elif contains_any(p, ["shield"]):
        draw.ellipse([x - 35, y - 45, x + 35, y + 45], fill=BLUE, outline=INK, width=4)
    elif contains_any(p, ["crown"]):
        pts = [(x - 35, y + 20), (x - 25, y - 20), (x, y + 5), (x + 25, y - 20), (x + 35, y + 20)]
        draw.polygon(pts, fill=GOLD, outline=INK)
    elif contains_any(p, ["torch", "fire"]):
        draw.rectangle([x - 6, y, x + 6, y + 55], fill=BROWN)
        draw.polygon([(x, y - 20), (x - 14, y + 6), (x, y + 18), (x + 14, y + 6)], fill=GOLD, outline=RED)
    elif contains_any(p, ["book", "tablet", "stone"]):
        draw.rectangle([x - 30, y - 35, x + 30, y + 35], fill=STONE, outline=INK, width=4)
        for i in range(-15, 20, 12):
            draw.line([x - 18, y + i, x + 18, y + i], fill=INK, width=2)
    elif contains_any(p, ["coin", "gold"]):
        draw.ellipse([x - 25, y - 25, x + 25, y + 25], fill=GOLD, outline=INK, width=3)
    elif contains_any(p, ["pyramid"]):
        draw.polygon([(x, y - 75), (x - 75, y + 45), (x + 75, y + 45)], fill=SAND, outline=INK)
    elif contains_any(p, ["ship", "boat"]):
        draw.polygon([(x - 80, y), (x + 80, y), (x + 50, y + 35), (x - 55, y + 35)], fill=BROWN, outline=INK)
        draw.line([x, y, x, y - 90], fill=INK, width=5)
        draw.polygon([(x, y - 80), (x + 45, y - 35), (x, y - 35)], fill=PAPER, outline=INK)
    elif contains_any(p, ["temple", "gate"]):
        draw.rectangle([x - 70, y - 50, x + 70, y + 50], fill=theme_color_from_prop(p), outline=INK, width=4)
        draw.polygon([(x - 85, y - 50), (x, y - 95), (x + 85, y - 50)], fill=GOLD, outline=INK)
    else:
        # Generic box/marker so unknown props still leave a visual cue.
        draw.rectangle([x - 28, y - 28, x + 28, y + 28], fill=STONE, outline=INK, width=3)


def theme_color_from_prop(_: str) -> tuple[int, int, int]:
    return (171, 92, 65)


def make_scene(scene: dict[str, Any], era: str, output_path: Path) -> None:
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), PAPER)
    draw = ImageDraw.Draw(img)
    theme = era_theme(era, scene.get("setting", ""))
    draw_background(draw, theme, scene.get("setting", ""))

    chars = scene.get("characters") or ["historical figure"]
    # Keep scenes readable: max 4 core figures.
    char_positions = [(260, 600), (520, 600), (820, 600), (1060, 600)]
    actions = scene.get("action", "")
    for idx, label in enumerate(chars[:4]):
        x, y = char_positions[idx]
        if len(chars) > 2:
            scale = 0.82
        else:
            scale = 1.0
        draw_stickman(draw, x, y, scale, str(label), era, actions, flip=(idx % 2 == 1))

    props = scene.get("props") or []
    prop_positions = [(1050, 520), (180, 500), (640, 430)]
    for idx, prop in enumerate(props[:3]):
        x, y = prop_positions[idx]
        draw_prop(draw, x, y, str(prop), 1.0)

    # A small scene label is useful visually, but never burns narration/subtitles.
    label = normalize_spaces(scene.get("setting", ""))[:48]
    if label:
        draw.rounded_rectangle([40, 40, 40 + min(580, 25 * len(label) + 40), 92], radius=16, fill=(255, 255, 255), outline=INK, width=3)
        draw.text((58, 53), label, font=FONT_28, fill=INK)

    # Tiny decorative hand-drawn dots/lines to keep the frame from feeling too sterile.
    random.seed(scene.get("id", 0) * 7919)
    for _ in range(22):
        x = random.randint(60, 1210)
        y = random.randint(105, 440)
        r = random.choice([2, 3, 4])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(140, 140, 140))

    img.save(output_path, quality=92)


def render_scenes(script: dict[str, Any]) -> None:
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    for idx, scene in enumerate(script["scenes"], 1):
        out = SCENE_DIR / f"scene_{idx:03d}.jpg"
        if out.exists() and out.stat().st_size > 5000:
            print(f"[SCENE] Reusing {out.name}")
            continue
        make_scene(scene, script["era"], out)
    checkpoint("scenes_complete")


# ---------------------------------------------------------------------------
# Thumbnail
# ---------------------------------------------------------------------------
def render_thumbnail(script: dict[str, Any], title: str, variant: int, out_path: Path) -> None:
    thumb = script.get("thumbnail", {})
    image = Image.new("RGB", (1280, 720), PAPER)
    draw = ImageDraw.Draw(image)
    theme = era_theme(script.get("era", ""), thumb.get("subject", ""))
    draw_background(draw, theme, thumb.get("subject", ""))

    composition = str(thumb.get("composition", "left_subject_right_prop"))
    subject_x = 360 if "left" in composition else 900 if "right" in composition else 640
    prop_x = 930 if subject_x < 600 else 350
    if variant == 2:
        subject_x, prop_x = prop_x, subject_x
    elif variant == 3:
        subject_x, prop_x = 640, 640

    draw_stickman(
        draw,
        subject_x,
        610,
        1.65,
        str(thumb.get("subject", "historical figure")),
        script.get("era", ""),
        str(thumb.get("emotion", "surprised and curious")),
        flip=subject_x > prop_x,
    )
    draw_prop(draw, prop_x, 500, str(thumb.get("supporting_prop", "scroll")), 1.8)

    headline = normalize_spaces(str(thumb.get("headline", "WHY DID THIS HAPPEN?"))).upper()
    lines = text_wrap_for_image(headline, 14)
    x = 60 if variant != 2 else 720
    y = 55
    for line in lines[:3]:
        draw.rounded_rectangle([x - 10, y - 5, min(1230, x + 600), y + 80], radius=18, fill=BLACK)
        draw.text((x + 8, y + 8), line, font=FONT_64, fill=WHITE)
        y += 86

    draw.text((50, 655), script.get("era", "History"), font=FONT_24, fill=INK)
    draw.rectangle([10, 10, 1270, 710], outline=INK, width=6)
    image.save(out_path, quality=95)


def make_thumbnail(script: dict[str, Any], title: str) -> Path:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    preferred = int(script.get("thumbnail", {}).get("preferred_variant", 1) or 1)
    preferred = min(3, max(1, preferred))
    outputs: list[Path] = []
    for variant in (1, 2, 3):
        out = THUMB_DIR / f"thumbnail_v{variant}.jpg"
        render_thumbnail(script, title, variant, out)
        outputs.append(out)
    chosen = outputs[preferred - 1]
    final = OUTPUT_DIR / "thumbnail.jpg"
    final.write_bytes(chosen.read_bytes())
    print(f"[THUMBNAIL] chosen variant {preferred}: {final}")
    return final


# ---------------------------------------------------------------------------
# SEO
# ---------------------------------------------------------------------------
SEO_PROMPT = """
You are the YouTube packaging editor for a history storytelling channel.

The video answers a specific historical curiosity question.
Create metadata that is discoverable without sounding like spam.

Rules:
- One primary title under 70 characters. Natural curiosity, no fake claims.
- Two alternate titles.
- Description: the first two lines should clearly explain the question and why the story matters.
  Then a concise spoiler-light summary, followed by a Sources section using the supplied sources.
- Tags: 12-15 relevant terms. Tags are secondary; do not stuff unrelated keywords.
- Thumbnail headline: 2-5 words that complement the title instead of repeating it.

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


def build_seo(topic: dict[str, Any], script: dict[str, Any], research: str) -> dict[str, Any]:
    prompt = (
        SEO_PROMPT
        + "\n\nCENTRAL QUESTION:\n"
        + topic["question"]
        + "\n\nSCRIPT:\n"
        + "\n\n".join(scene["narration"] for scene in script["scenes"])
        + "\n\nRESEARCH SOURCES / NOTES:\n"
        + research[-12000:]
    )
    seo = groq_json(
        GROQ_LIGHT_MODEL,
        [{"role": "user", "content": prompt}],
        max_completion_tokens=3000,
        temperature=0.55,
    )
    if not seo.get("title") or not seo.get("description") or not seo.get("tags"):
        raise RuntimeError("SEO output is incomplete.")
    seo["title"] = normalize_spaces(str(seo["title"]))[:100]
    seo["tags"] = [normalize_spaces(str(x)) for x in seo.get("tags", []) if normalize_spaces(str(x))]
    seo["tags"] = seo["tags"][:15]
    return seo


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
            lines.append(f"duration {durations[idx]:.4f}")
    if durations is not None and paths:
        lines.append(f"file '{paths[-1].resolve().as_posix()}'")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_video(script: dict[str, Any], out_path: Path) -> float:
    scenes = script["scenes"]
    image_paths = [SCENE_DIR / f"scene_{i:03d}.jpg" for i in range(1, len(scenes) + 1)]
    audio_paths = [AUDIO_DIR / f"scene_{i:03d}.wav" for i in range(1, len(scenes) + 1)]
    for path in image_paths + audio_paths:
        if not path.exists():
            raise RuntimeError(f"Missing render asset: {path}")

    durations = [audio_duration(path) for path in audio_paths]
    total = sum(durations)

    video_list = WORK_DIR / "video_concat.txt"
    build_concat_file(image_paths, video_list, durations)
    video_silent = OUTPUT_DIR / "video_silent.mp4"
    run_cmd(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(video_list),
            "-vf", f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-r", str(VIDEO_FPS),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(video_silent),
        ],
        "render static scene video",
    )

    audio_list = WORK_DIR / "audio_concat.txt"
    build_concat_file(audio_paths, audio_list)
    full_audio = OUTPUT_DIR / "full_audio.wav"
    run_cmd(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(audio_list),
            "-ar", str(AUDIO_SR), "-ac", "1", "-c:a", "pcm_s16le",
            str(full_audio),
        ],
        "concatenate narration",
    )

    mixed_audio = OUTPUT_DIR / "full_audio_mixed.m4a"
    if MUSIC_PATH.exists():
        run_cmd(
            [
                "ffmpeg", "-y",
                "-i", str(full_audio),
                "-stream_loop", "-1", "-i", str(MUSIC_PATH),
                "-filter_complex",
                "[0:a]highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11[n];"
                "[1:a]volume=0.06[m];[n][m]amix=inputs=2:duration=first:dropout_transition=2[a]",
                "-map", "[a]", "-c:a", "aac", "-b:a", "192k",
                str(mixed_audio),
            ],
            "mix background music",
        )
    else:
        run_cmd(
            [
                "ffmpeg", "-y", "-i", str(full_audio),
                "-af", "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11",
                "-c:a", "aac", "-b:a", "192k",
                str(mixed_audio),
            ],
            "normalize narration",
        )

    run_cmd(
        [
            "ffmpeg", "-y",
            "-i", str(video_silent),
            "-i", str(mixed_audio),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path),
        ],
        "mux final video",
    )
    print(f"[VIDEO] ready: {out_path} (~{total / 60:.1f} min)")
    checkpoint("video_complete", duration_seconds=round(total, 2))
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
    print(f"=== {CHANNEL_NAME} / Motive Unknown v2 ===")
    print(f"MODE={mode} | KOKORO_VOICE={KOKORO_VOICE} | SPEED={KOKORO_SPEED}")

    if mode == "voice_test":
        voice_test()
        return

    require_secret("GROQ_API_KEY")
    require_secret("YOUTUBE_TOKEN_JSON")
    require_secret("YOUTUBE_CLIENT_SECRET_JSON")

    checkpoint("starting")
    history = load_history()

    # Stage 1: curiosity question
    topic = choose_topic(history)
    atomic_write_json(TOPIC_PATH, topic)
    checkpoint("topic_complete", question=topic["question"])

    # Stage 2: research
    research = research_topic(topic)
    RESEARCH_PATH.write_text(research, encoding="utf-8")
    checkpoint("research_complete")

    # Stage 3: story architecture
    story = build_story_plan(topic, research)
    atomic_write_json(STORY_PATH, story)
    checkpoint("story_plan_complete")

    # Stage 4: final narration + scene plan
    script = write_script(topic, research, story)
    atomic_write_json(SCRIPT_PATH, script)
    checkpoint("script_complete", scene_count=len(script["scenes"]), words=sum(count_words(s["narration"]) for s in script["scenes"]))

    # Stage 5: local audio
    generate_voiceovers(script)

    # Stage 6: local illustrated scenes
    render_scenes(script)

    # Stage 7: SEO
    seo = build_seo(topic, script, research)
    atomic_write_json(SEO_PATH, seo)
    checkpoint("seo_complete", title=seo["title"])

    # Stage 8: dedicated thumbnail
    thumb = make_thumbnail(script, seo["title"])

    # Stage 9: final video
    video_path = OUTPUT_DIR / "final_video.mp4"
    build_video(script, video_path)

    # Stage 10: upload
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
    checkpoint("complete", video_id=video_id, title=seo["title"])
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "voice_test"], default="full")
    args = parser.parse_args()
    main(args.mode)
