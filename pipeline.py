"""
Motive Unknown — automated video pipeline (puppet-animation format).
Runs unattended on GitHub Actions. No interactive input anywhere.

Required GitHub Secrets (Settings -> Secrets and variables -> Actions):
  NVIDIA_NIM_API_KEY    - your NVIDIA NIM key
  YOUTUBE_TOKEN_JSON    - the full contents of your youtube_token.json file
  YOUTUBE_CLIENT_SECRET_JSON - the full contents of your client_secret_....json file

Required repo contents:
  assets/  - the cleaned character PNGs (from motive_unknown_clean_assets.zip)
"""

import os
import re
import time
import json
import math
import asyncio
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------
# 0. Load secrets from environment (GitHub injects these at runtime)
# ---------------------------------------------------------------------
# REVERTED from Gemini back to NVIDIA NIM — Gemini's free-tier quota kept
# getting fully exhausted (not just per-minute, since even 60s waits after
# a 429 kept failing again), and no amount of pacing fixes a quota that's
# actually used up. NIM is slower per call but is the one provider that's
# proven to run a full crew to completion without this wall.
NVIDIA_KEY = os.environ["NVIDIA_NIM_API_KEY"]

with open("youtube_token.json", "w") as f:
    f.write(os.environ["YOUTUBE_TOKEN_JSON"])
with open("client_secret.json", "w") as f:
    f.write(os.environ["YOUTUBE_CLIENT_SECRET_JSON"])

print("Secrets loaded.")

# ---------------------------------------------------------------------
# 1. LLM connection (NVIDIA NIM)
# ---------------------------------------------------------------------
import requests
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import tool

llm = LLM(
    model="openai/nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=NVIDIA_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    timeout=300,
    max_retries=5,
)


def call_with_retry(llm_obj, prompt, attempts=5, base_delay=10):
    last_error = None
    for i in range(attempts):
        try:
            return llm_obj.call(prompt)
        except Exception as e:
            last_error = e
            wait = base_delay * (i + 1)
            print(f"LLM call failed (attempt {i+1}/{attempts}): {e}\nRetrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {attempts} attempts: {last_error}")


test = call_with_retry(llm, "Reply with exactly one word: OK")
print("LLM connection test:", test)

# ---------------------------------------------------------------------
# 2. Search tools
#
# DDGS (duckduckgo_search) is gone. It scrapes DDG's HTML endpoint with no
# official API, and GitHub Actions runner IPs get bot-challenged/rate-limited
# on it constantly — which doesn't just cost the 3 retries in this function,
# it also makes the CrewAI agent re-issue the search several more times on
# its own, burning full LLM calls each time. That combo is most of why a run
# can still be stuck on an early task after 90 minutes.
#
# Replaced with two genuinely free (not free-tier-capped) sources:
#   - SearXNG: a self-hosted metasearch engine (Google/Bing/DDG/70+ engines
#     merged). No API key, no rate limit, no monthly fee, because you're
#     running it yourself. Needs a SearXNG instance reachable at
#     SEARXNG_URL (see the GitHub Actions service block in
#     searxng-github-actions.yml). It also needs `search.formats: [html, json]`
#     enabled in SearXNG's settings.yml — JSON output is off by default.
#   - Wikipedia's REST/API: official, free, unlimited for reasonable use, no
#     key. Since every topic here is a specific Ancient Egypt / Ancient Rome
#     figure or event, Wikipedia is a much more reliable primary source for
#     the Researcher than search snippets are — use SearXNG mainly for the
#     Topic Scout's "what's trending" angle instead.
# ---------------------------------------------------------------------
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")
WIKI_USER_AGENT = "MotiveUnknownBot/1.0 (automated video pipeline; contact: replace-with-your-email@example.com)"


@tool("Web Search")
def search_tool(query: str) -> str:
    """Searches the web via a self-hosted SearXNG instance and returns top results with titles, snippets, and links. Best for trending topics, angles, and general web context."""
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])[:6]
            if not results:
                return "No results found for this query. Try a different angle."
            formatted = []
            for r in results:
                formatted.append(f"{r.get('title','')}\n{r.get('content','')}\n{r.get('url','')}")
            return "\n\n".join(formatted)
        except Exception as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    return f"Search failed after 3 attempts ({last_error}). Proceed using general knowledge instead."


@tool("Wikipedia Lookup")
def wikipedia_tool(query: str) -> str:
    """Looks up a topic, figure, or event on Wikipedia and returns summaries of the top matching pages. Best for verifying specific historical facts, figures, and events — use this before falling back to general web search."""
    last_error = None
    for attempt in range(3):
        try:
            search_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": 3,
                },
                headers={"User-Agent": WIKI_USER_AGENT},
                timeout=15,
            )
            search_resp.raise_for_status()
            hits = search_resp.json().get("query", {}).get("search", [])
            if not hits:
                return "No Wikipedia results found for this query. Try a different angle or use Web Search instead."

            formatted = []
            for hit in hits:
                title = hit["title"]
                summary_resp = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
                    headers={"User-Agent": WIKI_USER_AGENT},
                    timeout=15,
                )
                if summary_resp.status_code != 200:
                    continue
                summary = summary_resp.json()
                extract = summary.get("extract", "")
                url = summary.get("content_urls", {}).get("desktop", {}).get("page", "")
                formatted.append(f"{title}\n{extract}\n{url}")

            if not formatted:
                return "Found matching titles but couldn't fetch summaries. Try Web Search instead."
            return "\n\n".join(formatted)
        except Exception as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    return f"Wikipedia lookup failed after 3 attempts ({last_error}). Proceed using general knowledge instead."


# ---------------------------------------------------------------------
# 3. Character roster — single source of truth, matches assets/ filenames.
#
# SCOPE NOTE: general "why does X happen" curiosity topics are on hold —
# there's no character art for non-historical content yet. Topic Scout
# below is scoped to Ancient Egypt / Ancient Rome only until that changes.
# ---------------------------------------------------------------------
CHARACTER_ROSTER = {
    "ancient_egypt": ["egyptian_commoner", "egyptian_soldier", "egyptian_royal"],
    "ancient_rome": ["roman_commoner", "roman_soldier", "roman_royal"],
}
VALID_EXPRESSIONS = ["neutral", "happy", "angry", "worried"]

# ---------------------------------------------------------------------
# 4. Agents
# ---------------------------------------------------------------------
topic_scout = Agent(
    role="Topic Scout",
    goal=(
        "Find a single, highly clickable video topic that is a specific historical "
        "story, event, or figure from Ancient Egypt or Ancient Rome — the only eras "
        "with character art available right now."
    ),
    backstory=(
        "You track trending searches, Reddit threads, and history content that performs "
        "well on YouTube. Your job is to spot a specific angle (not a broad topic) — a "
        "particular event, figure, or moment — that has strong curiosity-gap potential. "
        "You only pick topics from Ancient Egypt or Ancient Rome, since those are the "
        "only eras with character designs ready. You pick ONE topic and justify why it "
        "will perform well."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

researcher = Agent(
    role="Video Researcher",
    goal="Find the most surprising, well-sourced facts or story beats on the chosen historical topic",
    backstory=(
        "You're an obsessive researcher for a history storytelling channel. You dig up "
        "real, verifiable, surprising details and put them in the order they'd be told "
        "as a story. You avoid generic facts everyone already knows. You always check "
        "Wikipedia Lookup first for the topic and any figures/events it mentions — it's "
        "your primary, most reliable source. Only reach for Web Search when Wikipedia "
        "doesn't have enough detail on a specific angle."
    ),
    tools=[wikipedia_tool, search_tool],
    llm=llm,
    verbose=True,
)

scriptwriter = Agent(
    role="Scriptwriter",
    goal=(
        "Turn research into a 10-15 minute puppet-animation video script: a narrator "
        "tells the story while on-screen characters silently act along, breaking into "
        "their own dialogue or jokes only occasionally."
    ),
    backstory=(
        "You write for a 2D cutout/puppet animation history channel, similar in style "
        "to 'Chat History' and 'Peanut'. A narrator voice carries most of the runtime. "
        "The characters on screen are simple archetypes (commoner, soldier, royal) for "
        "the era of the story — they are not named historical figures, they're stand-ins "
        "acting out the story and occasionally cracking a joke or reacting to each other. "
        "You output ONLY valid JSON, nothing else — no preamble, no markdown code fences, "
        "no commentary before or after the JSON."
    ),
    llm=llm,
    verbose=True,
)

seo_specialist = Agent(
    role="YouTube SEO Specialist",
    goal="Generate a high-CTR title, description, and tag list for the video",
    backstory=(
        "You've studied thousands of high-performing history-channel uploads and know "
        "how to write curiosity-driven titles and keyword-rich descriptions."
    ),
    llm=llm,
    verbose=True,
)

print("Agents ready.")

# ---------------------------------------------------------------------
# 5. Tasks
# ---------------------------------------------------------------------
topic_task = Task(
    description=(
        "Search for a specific, under-covered historical story, event, or figure from "
        "Ancient Egypt or Ancient Rome ONLY — no other eras, and no non-historical "
        "curiosity topics. Pick ONE specific angle. State the topic clearly, name which "
        "era it's from (ancient_egypt or ancient_rome), and give 2-3 sentences on why it "
        "will perform well."
    ),
    expected_output="One clearly stated topic, its era, and a short justification.",
    agent=topic_scout,
)

research_task = Task(
    description=(
        "Using the topic chosen by the Topic Scout, research 8-12 specific, surprising, "
        "and verifiable facts or story beats, each with a one-line source or context. "
        "Put them roughly in the order they'd be told as a story."
    ),
    expected_output="A bullet list of 8-12 facts/story beats in story order, each with a short source note.",
    agent=researcher,
    context=[topic_task],
)

CHARACTER_LIST_TEXT = "\n".join(
    f"  {era}: {', '.join(names)}" for era, names in CHARACTER_ROSTER.items()
)

script_task = Task(
    description=(
        "Using the research, write a 10-15 minute puppet-animation video script as JSON.\n\n"
        "CRITICAL: This is a fully automated pipeline. There is no human available to "
        "answer questions or confirm details. Decide everything yourself and output the "
        "finished JSON directly, right now. Never ask a question, never say 'let me know', "
        "never wrap the JSON in markdown code fences, never write anything before or after "
        "the JSON object. Do NOT include a thinking process, analysis, reasoning section, or "
        "any notes about your approach — not even a short one. Your entire response must be "
        "nothing but the JSON object itself: the very first character you output must be '{' "
        "and the very last character must be '}'.\n\n"
        f"Only use these characters, matched to the story's era:\n{CHARACTER_LIST_TEXT}\n\n"
        "Output a single JSON object with this exact shape:\n"
        "{\n"
        '  "era": "ancient_egypt" | "ancient_rome",\n'
        '  "characters_used": ["<character_name>", ...],\n'
        '  "segments": [\n'
        "    {\n"
        '      "type": "narration",\n'
        '      "text": "<narrator line, 1-3 sentences>",\n'
        '      "on_screen": [{"character": "<name>", "expression": "<neutral|happy|angry|worried>"}]\n'
        "    },\n"
        "    {\n"
        '      "type": "dialogue",\n'
        '      "lines": [\n'
        '        {"speaker": "<character_name>", "text": "<line>", "expression": "<neutral|happy|angry|worried>"},\n'
        "        ...\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Most segments should be type 'narration' — this carries the story.\n"
        "- Use type 'dialogue' only occasionally (roughly every 4-8 narration segments), "
        "for a short back-and-forth exchange or joke between 2 characters already "
        "established as on_screen nearby.\n"
        "- Total narration + dialogue text combined should be roughly 1600-2400 words "
        "(this is a 10-15 minute voiceover at normal pacing).\n"
        "- Every character name used must come from the allowed list above, and must "
        "match the chosen era.\n"
        "- Open with a hook in the first narration segment.\n"
        "- Keep sentences short — this is read aloud by AI voiceover."
    ),
    expected_output=(
        "A single valid JSON object matching the schema above — nothing else, no "
        "markdown fences, no explanation text."
    ),
    agent=scriptwriter,
    context=[research_task],
)

seo_task = Task(
    description=(
        "Based on the script, write:\n"
        "1. Three title options (under 60 characters, curiosity-driven, no clickbait flags)\n"
        "2. A YouTube description (first 2 lines keyword-rich, then a short summary)\n"
        "3. A list of 15 relevant tags"
    ),
    expected_output="Titles, description, and tags clearly labeled.",
    agent=seo_specialist,
    context=[script_task],
)

print("Tasks ready.")

# ---------------------------------------------------------------------
# 6. Run the crew
# ---------------------------------------------------------------------
crew = Crew(
    agents=[topic_scout, researcher, scriptwriter, seo_specialist],
    tasks=[topic_task, research_task, script_task, seo_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
print("\n\n===== FINAL OUTPUT =====\n")
print(result)

# ---------------------------------------------------------------------
# 7. Parse + validate the Scriptwriter's JSON output
# ---------------------------------------------------------------------
def _extract_json_object(raw_text: str) -> dict:
    """Some models (this NVIDIA Nemotron model included, in practice) write
    out a 'thinking process' before the real answer no matter how firmly
    you tell them not to — and that reasoning text can itself contain
    brace-like snippets ('keys: {"era", "segments"}') that break a naive
    first-brace/last-brace slice. This scans for every *balanced* {...}
    block in the text and tries them from LAST to FIRST (the real answer
    comes after the reasoning, not before it), returning the first one
    that both parses as JSON and actually looks like our script shape."""
    candidates = []
    stack = []
    start = None
    for i, ch in enumerate(raw_text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(raw_text[start:i + 1])
                    start = None

    last_error = None
    for cand in reversed(candidates):
        try:
            data = json.loads(cand)
            if isinstance(data, dict) and "segments" in data:
                return data
        except json.JSONDecodeError as e:
            last_error = e
            continue

    raise RuntimeError(
        f"No valid JSON object with a 'segments' key found anywhere in the output "
        f"({len(candidates)} brace-balanced candidate(s) tried, last parse error: {last_error}). "
        f"Raw output:\n{raw_text[:1500]}"
    )


def parse_script_json(raw_text: str) -> dict:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    data = _extract_json_object(text)

    era = data.get("era")
    if era not in CHARACTER_ROSTER:
        raise RuntimeError(f"Script has invalid/missing era: {era!r}")

    allowed_for_era = set(CHARACTER_ROSTER[era])
    segments = data.get("segments")
    if not isinstance(segments, list) or len(segments) < 4:
        raise RuntimeError("Script has too few segments (or 'segments' missing/not a list).")

    total_words = 0
    for i, seg in enumerate(segments):
        seg_type = seg.get("type")
        if seg_type == "narration":
            text_val = seg.get("text", "")
            total_words += len(text_val.split())
            for char in seg.get("on_screen", []):
                _validate_character(char, allowed_for_era, i)
        elif seg_type == "dialogue":
            lines = seg.get("lines", [])
            if not lines:
                raise RuntimeError(f"Segment {i} is type 'dialogue' but has no lines.")
            for line in lines:
                total_words += len(line.get("text", "").split())
                line["character"] = line.get("speaker")  # so _validate_character can read it uniformly
                _validate_character(line, allowed_for_era, i)
        else:
            raise RuntimeError(f"Segment {i} has invalid type: {seg_type!r}")

    if total_words < 1000 or total_words > 3200:
        raise RuntimeError(
            f"Script word count ({total_words}) is way outside the expected 1600-2400 "
            "word range — likely a truncated or malformed generation."
        )

    print(f"Script validated: era={era}, {len(segments)} segments, ~{total_words} words.")
    return data


def _validate_character(char_entry: dict, allowed_for_era: set, seg_index: int, expr_key: str = "expression"):
    """Character name has no safe fallback (there's no art for a name that
    doesn't exist), so that still fails hard. An unrecognized expression
    DOES have a safe fallback (neutral) — so rather than throw away a full
    ~25-30 minute crew run over one made-up word like 'proud', we just warn
    and downgrade it to neutral, and keep going."""
    name = char_entry.get("character")
    if name not in allowed_for_era:
        raise RuntimeError(
            f"Segment {seg_index} uses character {name!r}, which isn't in this era's "
            f"roster ({sorted(allowed_for_era)})."
        )
    expr = char_entry.get(expr_key)
    if expr not in VALID_EXPRESSIONS:
        print(f"WARNING: segment {seg_index}, character {name!r} used unrecognized "
              f"expression {expr!r} — falling back to 'neutral'.")
        char_entry[expr_key] = "neutral"


def flatten_script_to_text(parsed_script: dict) -> str:
    parts = []
    for seg in parsed_script["segments"]:
        if seg["type"] == "narration":
            parts.append(seg["text"])
        else:
            for line in seg["lines"]:
                parts.append(f'{line["speaker"]}: {line["text"]}')
    return "\n".join(parts)


raw_task_out = getattr(script_task.output, "raw", str(script_task.output))
parsed_script = parse_script_json(raw_task_out)

# ---------------------------------------------------------------------
# 8. Character assembly config
# ---------------------------------------------------------------------
import edge_tts

ASSET_DIR = "assets"

CHARACTER_FILES = {
    "egyptian_commoner": "egyptian_commoner.png",
    "egyptian_soldier": "egyptian_soldier.png",
    "egyptian_royal": "egyptian_royal.png",
    "roman_commoner": "roman_commoner.png",
    "roman_soldier": "roman_soldier.png",
    "roman_royal": "roman_royal.png",
}
EYE_FILES = {
    "neutral": "eyes_neutral.png",
    "happy": "eyes_happy.png",
    "angry": "eyes_angry.png",
    "worried": "eyes_neutral.png",  # no distinct "worried" eyes drawn yet — falls back to neutral
}
MOUTH_FILES = {
    "closed": "mouth_closed.png",
    "half": "mouth_half_open.png",
    "open": "mouth_fully_open.png",
}
NARRATOR_VOICE = "en-US-GuyNeural"
VOICE_MAP = {
    "egyptian_commoner": "en-US-DavisNeural",
    "egyptian_soldier": "en-US-TonyNeural",
    "egyptian_royal": "en-US-JennyNeural",
    "roman_commoner": "en-US-EricNeural",
    "roman_soldier": "en-GB-RyanNeural",
    "roman_royal": "en-US-AriaNeural",
}
# FIRST-PASS eyes/mouth placement, as fraction of (width, height) — nudge
# per character once you've seen a real render; a single universal ratio
# was tested and does NOT land well on every head shape.
REGISTRATION = {
    "egyptian_commoner": {"eyes": (0.50, 0.13), "mouth": (0.50, 0.22)},
    "egyptian_soldier":  {"eyes": (0.50, 0.12), "mouth": (0.50, 0.21)},
    "egyptian_royal":    {"eyes": (0.50, 0.14), "mouth": (0.50, 0.23)},
    "roman_commoner":    {"eyes": (0.50, 0.13), "mouth": (0.50, 0.22)},
    "roman_soldier":     {"eyes": (0.50, 0.12), "mouth": (0.50, 0.21)},
    "roman_royal":       {"eyes": (0.50, 0.13), "mouth": (0.50, 0.22)},
}

FPS = 12
VIDEO_W, VIDEO_H = 1280, 720
SCENE_DIR = "scene_frames"

# ---------------------------------------------------------------------
# 9. Voiceover — one file per narration block / dialogue line
# ---------------------------------------------------------------------
def tts_to_file(text: str, voice: str, filename: str):
    async def _run():
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(filename)
    asyncio.run(_run())


def generate_all_voiceovers(parsed_script: dict, out_dir: str = "audio"):
    os.makedirs(out_dir, exist_ok=True)
    clip_index = 0
    for seg in parsed_script["segments"]:
        if seg["type"] == "narration":
            fname = os.path.join(out_dir, f"clip_{clip_index:04d}_narration.mp3")
            tts_to_file(seg["text"], NARRATOR_VOICE, fname)
            seg["audio_file"] = fname
            clip_index += 1
        else:
            for line in seg["lines"]:
                voice = VOICE_MAP.get(line["speaker"], NARRATOR_VOICE)
                fname = os.path.join(out_dir, f"clip_{clip_index:04d}_{line['speaker']}.mp3")
                tts_to_file(line["text"], voice, fname)
                line["audio_file"] = fname
                clip_index += 1
    print(f"Generated {clip_index} voiceover clips in {out_dir}/")
    return parsed_script


# ---------------------------------------------------------------------
# 10. Amplitude envelope -> cheap lip-sync
# ---------------------------------------------------------------------
def get_amplitude_envelope(mp3_path: str, fps: int = FPS):
    raw_path = mp3_path + ".pcm"
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-f", "s16le", "-ac", "1", "-ar", "16000", raw_path],
        check=True, capture_output=True,
    )
    samples = np.fromfile(raw_path, dtype=np.int16).astype(np.float32) / 32768.0
    os.remove(raw_path)

    samples_per_frame = max(1, int(16000 / fps))
    n_frames = max(1, math.ceil(len(samples) / samples_per_frame))
    envelope = np.zeros(n_frames)
    for i in range(n_frames):
        chunk = samples[i * samples_per_frame:(i + 1) * samples_per_frame]
        if len(chunk):
            envelope[i] = np.sqrt(np.mean(chunk ** 2))

    peak = envelope.max() if envelope.max() > 0 else 1.0
    return envelope / peak, n_frames / fps


def amplitude_to_mouth(a: float) -> str:
    if a < 0.15:
        return "closed"
    if a < 0.55:
        return "half"
    return "open"


# ---------------------------------------------------------------------
# 11. Character compositing + frame rendering
# ---------------------------------------------------------------------
_asset_cache = {}


def _load(name: str) -> Image.Image:
    if name not in _asset_cache:
        _asset_cache[name] = Image.open(os.path.join(ASSET_DIR, name)).convert("RGBA")
    return _asset_cache[name]


def compose_character(character: str, expression: str, mouth_key: str = "closed") -> Image.Image:
    body = _load(CHARACTER_FILES[character])
    eyes = _load(EYE_FILES[expression])
    mouth = _load(MOUTH_FILES[mouth_key])
    reg = REGISTRATION[character]

    canvas = body.copy()
    ex_ratio, ey_ratio = reg["eyes"]
    mx_ratio, my_ratio = reg["mouth"]
    ex = int(body.width * ex_ratio) - eyes.width // 2
    ey = int(body.height * ey_ratio) - eyes.height // 2
    mx = int(body.width * mx_ratio) - mouth.width // 2
    my = int(body.height * my_ratio) - mouth.height // 2

    canvas.alpha_composite(eyes, (ex, ey))
    canvas.alpha_composite(mouth, (mx, my))
    return canvas


def _render_frame(on_screen: list, speaking_amp: dict) -> Image.Image:
    canvas = Image.new("RGBA", (VIDEO_W, VIDEO_H), (235, 225, 200, 255))
    n = max(1, len(on_screen))
    slot_w = VIDEO_W // (n + 1)
    for i, entry in enumerate(on_screen):
        char = entry["character"]
        expr = entry.get("expression", "neutral")
        amp = speaking_amp.get(char, 0.0)
        mouth_key = amplitude_to_mouth(amp)
        sprite = compose_character(char, expr, mouth_key)

        target_h = int(VIDEO_H * 0.6)
        scale = target_h / sprite.height
        sprite = sprite.resize((int(sprite.width * scale), target_h))

        x = slot_w * (i + 1) - sprite.width // 2
        y = VIDEO_H - sprite.height - 40
        canvas.alpha_composite(sprite, (x, y))

    return canvas.convert("RGB")


# ---------------------------------------------------------------------
# 12. Full video assembly — CACHED FRAMES, NOT ONE FILE PER 1/12s
#
# WHY THIS CHANGED: the previous version wrote one PNG per video frame for
# the ENTIRE video — 7,000-11,000 individual composited files for a 10-15
# minute video at 12fps. That's almost certainly what actually caused the
# 90-minute timeouts (the crew itself finishes in ~25-30 min even on a slow
# run — the rest of the time was unaccounted for, and this was the only
# part of the pipeline that scales with video LENGTH rather than segment
# COUNT). A puppet's mouth only changes a handful of times per second, so
# most of those frames were exact duplicates of the one before them.
#
# Fix: render each unique (character, expression, mouth-shape) combination
# ONCE, cache it, and tell ffmpeg to hold that single image for however
# long it's needed via the concat demuxer's `duration` directive — instead
# of writing the same image to disk over and over. Verified this actually
# works (correct total duration, correct visual output) before shipping it.
# ---------------------------------------------------------------------
_frame_cache = {}


def _get_or_render_frame(state: tuple) -> str:
    """state = tuple of (character, expression, mouth_key) tuples, one per
    on-screen character. Returns a file path, rendering + saving only the
    first time a given state is ever needed."""
    if state in _frame_cache:
        return _frame_cache[state]
    on_screen = [{"character": c, "expression": e} for c, e, _m in state]
    speaking_amp = {}  # unused now — we pass mouth_key directly below instead
    canvas = Image.new("RGBA", (VIDEO_W, VIDEO_H), (235, 225, 200, 255))
    n = max(1, len(state))
    slot_w = VIDEO_W // (n + 1)
    for i, (char, expr, mouth_key) in enumerate(state):
        sprite = compose_character(char, expr, mouth_key)
        target_h = int(VIDEO_H * 0.6)
        scale = target_h / sprite.height
        sprite = sprite.resize((int(sprite.width * scale), target_h))
        x = slot_w * (i + 1) - sprite.width // 2
        y = VIDEO_H - sprite.height - 40
        canvas.alpha_composite(sprite, (x, y))

    os.makedirs("unique_frames", exist_ok=True)
    path = f"unique_frames/{abs(hash(state))}.png"
    canvas.convert("RGB").save(path)
    _frame_cache[state] = path
    return path


def _get_audio_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _build_render_plan(parsed_script: dict):
    """Walks the script once, producing (image_path, hold_duration) pairs
    for ffmpeg's concat demuxer, plus the ordered list of audio files to
    concatenate. No per-video-frame files are written here at all — only
    as many unique images as the script actually needs."""
    plan = []
    audio_files_in_order = []

    for seg in parsed_script["segments"]:
        if seg["type"] == "narration":
            on_screen = seg.get("on_screen", [])
            # Narrator isn't a rendered character, and on-screen characters
            # stay silent/closed-mouth during narration — so this is ONE
            # static image for the whole segment, not per-frame amplitude
            # analysis at all. Getting duration via ffprobe is much cheaper
            # than decoding the full PCM waveform for something we don't
            # even use here.
            state = tuple(sorted((c["character"], c.get("expression", "neutral"), "closed") for c in on_screen))
            duration = _get_audio_duration(seg["audio_file"])
            plan.append((_get_or_render_frame(state), duration))
            audio_files_in_order.append(seg["audio_file"])
        else:
            speakers = [{"character": l["speaker"], "expression": l["expression"]} for l in seg["lines"]]
            for line in seg["lines"]:
                # Dialogue DOES need the real amplitude envelope, since the
                # speaking character's mouth actually needs to move — but
                # we collapse consecutive identical mouth-shapes into ONE
                # held image instead of one file per 1/12s slot.
                envelope, duration = get_amplitude_envelope(line["audio_file"])
                mouth_keys = [amplitude_to_mouth(a) for a in envelope]
                frame_dur = duration / len(mouth_keys) if mouth_keys else duration
                i = 0
                while i < len(mouth_keys):
                    j = i
                    while j + 1 < len(mouth_keys) and mouth_keys[j + 1] == mouth_keys[i]:
                        j += 1
                    hold_duration = (j - i + 1) * frame_dur
                    state = tuple(sorted(
                        (s["character"], s.get("expression", "neutral"),
                         mouth_keys[i] if s["character"] == line["speaker"] else "closed")
                        for s in speakers
                    ))
                    plan.append((_get_or_render_frame(state), hold_duration))
                    i = j + 1
                audio_files_in_order.append(line["audio_file"])

    return plan, audio_files_in_order


def build_video(parsed_script: dict, out_path: str = "final_video.mp4"):
    plan, audio_files_in_order = _build_render_plan(parsed_script)
    print(f"Render plan: {len(plan)} timeline entries, only {len(_frame_cache)} unique images "
          f"actually rendered (this is the number that used to be 7,000-11,000).")

    video_concat_path = "video_concat_list.txt"
    with open(video_concat_path, "w") as f:
        for path, dur in plan:
            f.write(f"file '{os.path.abspath(path)}'\n")
            f.write(f"duration {dur:.3f}\n")
        if plan:
            # ffmpeg's concat demuxer quirk: the LAST file's `duration` line
            # is ignored unless the file is listed once more after it.
            f.write(f"file '{os.path.abspath(plan[-1][0])}'\n")

    audio_concat_path = "audio_concat_list.txt"
    with open(audio_concat_path, "w") as f:
        for path in audio_files_in_order:
            f.write(f"file '{os.path.abspath(path)}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", audio_concat_path,
         "-c", "copy", "full_audio.mp3"],
        check=True, capture_output=True,
    )

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", video_concat_path,
         "-i", "full_audio.mp3",
         "-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2,fps={FPS}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", out_path],
        check=True, capture_output=True,
    )
    total_duration = sum(d for _, d in plan)
    print(f"Video ready: {out_path} (~{total_duration:.0f}s of content, {len(_frame_cache)} unique frames)")
    return out_path


# ---------------------------------------------------------------------
# 13. Thumbnail
# ---------------------------------------------------------------------
def generate_thumbnail(parsed_script: dict, title: str, out_path: str = "thumbnail.jpg"):
    on_screen = []
    for seg in parsed_script["segments"]:
        candidates = seg.get("on_screen") if seg["type"] == "narration" else \
            [{"character": l["speaker"], "expression": l["expression"]} for l in seg["lines"]]
        if candidates:
            on_screen = candidates
            break

    frame = _render_frame(on_screen, speaking_amp={})
    draw = ImageDraw.Draw(frame)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
    except Exception:
        font = ImageFont.load_default()

    words = title.upper().split(" ")
    y = 30
    for word in words:
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
            draw.text((30 + dx, y + dy), word, font=font, fill="black")
        draw.text((30, y), word, font=font, fill="white")
        y += 100

    frame.save(out_path, quality=90)
    print(f"Thumbnail ready: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# 14. Upload
# ---------------------------------------------------------------------
def extract_seo_title(text):
    match = re.search(r"1\.\s*[\"\u201c]?(.+?)[\"\u201d]?\s*(?:\(|$)", text)
    return match.group(1).strip() if match else "Automated Video"


def upload_video(video_path: str, thumbnail_path: str, title: str, description: str, tags: list):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    credentials = Credentials.from_authorized_user_file("youtube_token.json", SCOPES)
    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            raise RuntimeError(
                "YouTube token is invalid and can't be refreshed automatically. "
                "Re-run the one-time manual login step to generate a fresh youtube_token.json."
            )

    youtube = build("youtube", "v3", credentials=credentials)

    request_body = {
        "snippet": {
            "title": title[:100],
            "description": description[:4900],
            "tags": tags,
            "categoryId": "24",
        },
        "status": {
            "privacyStatus": "private",  # keep this until you've watched a few uploads and trust it
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    response = youtube.videos().insert(part="snippet,status", body=request_body, media_body=media).execute()
    video_id = response["id"]
    print(f"Uploaded! https://youtu.be/{video_id}")

    try:
        youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_path)).execute()
        print("Thumbnail set.")
    except Exception as e:
        print(f"Thumbnail upload failed (likely phone verification not done yet): {e}")

    return video_id


# ---------------------------------------------------------------------
# 15. Run everything
# ---------------------------------------------------------------------
parsed_script = generate_all_voiceovers(parsed_script)
video_path = build_video(parsed_script)

seo_output = getattr(seo_task.output, "raw", str(seo_task.output))
video_title = extract_seo_title(seo_output)[:100]
thumb_path = generate_thumbnail(parsed_script, video_title)

upload_video(
    video_path, thumb_path, video_title, seo_output[:4900],
    tags=["history", parsed_script["era"].replace("_", " ")],
)

print("\nDone.")
