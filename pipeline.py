"""
Motive Unknown — automated video pipeline.
Runs unattended on GitHub Actions. No interactive input anywhere.

Required GitHub Secrets (Settings -> Secrets and variables -> Actions):
  NVIDIA_NIM_API_KEY   - your NVIDIA NIM key
  ELEVENLABS_API_KEY   - your ElevenLabs key
  YOUTUBE_TOKEN_JSON    - the full contents of your youtube_token.json file
  YOUTUBE_CLIENT_SECRET_JSON - the full contents of your client_secret_....json file
"""

import os
import re
import time
import json
from urllib.parse import quote

import requests as req
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------
# 0. Load secrets from environment (GitHub injects these at runtime)
# ---------------------------------------------------------------------
NVIDIA_KEY = os.environ["NVIDIA_NIM_API_KEY"]
# ElevenLabs no longer used for voiceover — switched to Edge TTS (free, no API key,
# no character limit). Left here only in case you ever want to switch back.

# Write the two YouTube auth files to disk from secrets, so the rest of
# the code can use them exactly like it did in Colab.
with open("youtube_token.json", "w") as f:
    f.write(os.environ["YOUTUBE_TOKEN_JSON"])
with open("client_secret.json", "w") as f:
    f.write(os.environ["YOUTUBE_CLIENT_SECRET_JSON"])

print("Secrets loaded.")

# ---------------------------------------------------------------------
# 1. LLM connection (NVIDIA NIM)
# ---------------------------------------------------------------------
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import tool
from duckduckgo_search import DDGS

# max_retries + longer timeout: NVIDIA's free tier occasionally times out or
# gets briefly overloaded. This makes CrewAI retry the call itself with
# backoff instead of letting one slow moment kill the whole run.
llm = LLM(
    model="openai/nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=NVIDIA_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    timeout=300,
    max_retries=5,
)

# Belt-and-suspenders: also manually retry the initial connection test itself,
# since a failure here should not silently kill the whole workflow before
# the crew even starts.
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
# 2. Search tool + agents
# ---------------------------------------------------------------------
@tool("Web Search")
def search_tool(query: str) -> str:
    """Searches the web using DuckDuckGo and returns top results with titles, snippets, and links."""
    last_error = None
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(keywords=query, max_results=6))
            if not results:
                return "No results found for this query. Try a different angle."
            formatted = []
            for r in results:
                formatted.append(f"{r.get('title','')}\n{r.get('body','')}\n{r.get('href','')}")
            return "\n\n".join(formatted)
        except Exception as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    return f"Search failed after 3 attempts ({last_error}). Proceed using general knowledge instead."


topic_scout = Agent(
    role="Topic Scout",
    goal="Find a single, highly clickable video topic in the dark psychology / true crime facts niche based on what's currently trending or searched",
    backstory=(
        "You track trending searches, Reddit threads, and news in true crime and "
        "psychology. Your job is to spot a specific angle (not a broad topic) that "
        "has strong curiosity-gap potential and hasn't been overdone. You pick ONE "
        "topic and justify why it will perform well."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

researcher = Agent(
    role="Video Researcher",
    goal="Find the most surprising, well-sourced facts on the chosen topic for a dark psychology / true crime facts YouTube channel",
    backstory=(
        "You're an obsessive researcher for a viral facts channel. You dig up real, "
        "verifiable, unsettling-but-true details that make people stop scrolling. "
        "You avoid generic facts everyone already knows."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

scriptwriter = Agent(
    role="Scriptwriter",
    goal="Turn research into a tight, high-retention video script optimized for a faceless AI-voiceover video",
    backstory=(
        "You write scripts for a channel that lives or dies by the first 5 seconds. "
        "You open with a curiosity-gap hook, keep sentences short for voiceover pacing, "
        "and escalate each fact into the next. You write both a 60-second Shorts cut AND "
        "a 6-8 minute long-form version, clearly labeled with explicit (0s-10s) timestamps for the Shorts portion."
    ),
    llm=llm,
    verbose=True,
)

seo_specialist = Agent(
    role="YouTube SEO Specialist",
    goal="Generate a high-CTR title, description, and tag list for the video",
    backstory=(
        "You've studied thousands of high-performing faceless channel uploads and know "
        "how to write curiosity-driven titles and keyword-rich descriptions."
    ),
    llm=llm,
    verbose=True,
)

print("Agents ready.")

# ---------------------------------------------------------------------
# 3. Tasks
# ---------------------------------------------------------------------
topic_task = Task(
    description=(
        "Search for trending or under-covered angles in the dark psychology / true crime "
        "facts niche right now. Pick ONE specific video topic (not broad — a specific angle, "
        "case, or phenomenon). State the topic clearly and give 2-3 sentences on why it will "
        "perform well right now."
    ),
    expected_output="One clearly stated topic, plus a short justification.",
    agent=topic_scout,
)

research_task = Task(
    description=(
        "Using the topic chosen by the Topic Scout, research 6-10 specific, surprising, and "
        "verifiable facts, each with a one-line source or context. Avoid anything generic."
    ),
    expected_output="A bullet list of 6-10 facts, each with a short source note.",
    agent=researcher,
    context=[topic_task],
)

script_task = Task(
    description=(
        "Using the research, write two scripts.\n\n"
        "CRITICAL: This is a fully automated pipeline. There is no human available "
        "to answer questions, confirm details, or pick from options. You must decide "
        "everything yourself and output the finished scripts directly, right now, in "
        "this response. Never ask a question. Never say 'let me know' or offer choices. "
        "Never describe what you are about to write — just write it.\n\n"
        "Write in short, punchy sentences for AI voiceover pacing.\n\n"
        "First, write a header line: SHORTS SCRIPT\n"
        "Then write the actual 60-second script as a numbered list of timed lines, "
        "for example:\n"
        "(0s-10s) Line of dialogue goes here.\n"
        "(10s-20s) Next line goes here.\n"
        "...continuing until roughly 150-170 words total, opening with a hard hook "
        "in the very first line.\n\n"
        "Then write a header line: LONG-FORM SCRIPT\n"
        "Then write the actual 900-1100 word long-form script as normal paragraphs "
        "(no timestamps needed here), opening with the same hook, escalating through "
        "the facts, ending on the most surprising one."
    ),
    expected_output=(
        "The literal finished text of both scripts under their header lines — "
        "not a description of what the scripts should contain, and not a question."
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
# 4. Run the crew (synchronous — no Colab event-loop workaround needed)
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
# 5. Voiceover (ElevenLabs)
# ---------------------------------------------------------------------
raw_task_out = getattr(script_task.output, "raw", str(script_task.output))


def extract_section(text, start_marker, end_marker=None):
    start = text.upper().find(start_marker.upper())
    if start == -1:
        return None
    start += len(start_marker)
    if end_marker:
        end = text.upper().find(end_marker.upper(), start)
        if end == -1:
            end = len(text)
    else:
        end = len(text)
    return text[start:end].strip(" :\n*#")


shorts_script = extract_section(raw_task_out, "SHORTS SCRIPT", "LONG-FORM SCRIPT")
longform_script = extract_section(raw_task_out, "LONG-FORM SCRIPT")

if not shorts_script:
    shorts_script = raw_task_out[:500]
if not longform_script:
    longform_script = raw_task_out[500:]

print("Shorts script length:", len(shorts_script), "characters")
print("Long-form script length:", len(longform_script), "characters")

# Safety check: if the model echoed instructions, asked a question, or hedged
# instead of writing an actual script, stop here with a clear error instead of
# silently uploading garbage.
red_flags = [
    "timestamp headers", "hard hook", "150-170 words", "900-1100 words", "do not repeat",
    "let me know", "i can suggest", "please provide", "would you like", "just let me know",
    "or i can", "which would you", "should i",
]
lowered = shorts_script.lower()
if any(flag in lowered for flag in red_flags) or "?" in shorts_script or len(shorts_script) < 50:
    raise RuntimeError(
        "The Scriptwriter agent appears to have asked a question or echoed instructions "
        "instead of writing an actual script. Raw output was:\n\n" + raw_task_out[:1000]
    )

import edge_tts
import asyncio

VOICE_ID = "en-US-GuyNeural"  # deep, natural male voice — fits true crime/dark psychology tone
# Other good options to try: "en-US-EricNeural" (calm), "en-GB-RyanNeural" (British), "en-US-AriaNeural" (female)


def generate_voiceover(text, filename):
    async def _run():
        communicate = edge_tts.Communicate(text=text, voice=VOICE_ID)
        await communicate.save(filename)

    asyncio.run(_run())
    size = os.path.getsize(filename)
    print(f"Saved {filename} ({size} bytes) using Edge TTS voice '{VOICE_ID}'")


generate_voiceover(shorts_script, "shorts_voiceover.mp3")
generate_voiceover(longform_script, "longform_voiceover.mp3")

# ---------------------------------------------------------------------
# 6. Video assembly (images + captions + voiceover)
# ---------------------------------------------------------------------
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip

segments = []
pattern = re.compile(r"\((\d+)s-(\d+)s\)\s*\n?(.*?)(?=\(\d+s-\d+s\)|\Z)", re.DOTALL)
for match in pattern.finditer(shorts_script):
    start, end, text = int(match.group(1)), int(match.group(2)), match.group(3).strip()
    text = re.sub(r"\s+", " ", text)
    if text:
        segments.append({"start": start, "end": end, "text": text})

if not segments:
    print("Timestamp patterns not detected in script. Falling back to auto-chunking...")
    raw_lines = [l.strip() for l in shorts_script.splitlines() if l.strip() and not l.startswith("#")]
    time_per_line = 5
    for idx, line in enumerate(raw_lines[:12]):
        segments.append({"start": idx * time_per_line, "end": (idx + 1) * time_per_line, "text": line})

print(f"Found {len(segments)} segments.")

audio_clip = AudioFileClip("shorts_voiceover.mp3")
actual_duration = audio_clip.duration
script_total = segments[-1]["end"] if segments else 1
scale = actual_duration / script_total if script_total > 0 else 1
for seg in segments:
    seg["duration"] = (seg["end"] - seg["start"]) * scale
print(f"Voiceover length: {actual_duration:.1f}s — scaling segment timing by {scale:.2f}x to match.")

VIDEO_W, VIDEO_H = 1080, 1920

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
except Exception:
    font = ImageFont.load_default()


def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines, current = [], ""
    for w in words:
        test = (current + " " + w).strip()
        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


frame_paths = []
for i, seg in enumerate(segments):
    prompt = f"cinematic true crime documentary style, dark moody lighting, {seg['text'][:120]}"
    url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width={VIDEO_W}&height={VIDEO_H}&nologo=true"

    img = None
    for attempt in range(3):
        try:
            resp = req.get(url, timeout=45)
            resp.raise_for_status()
            with open(f"frame_{i}.jpg", "wb") as f:
                f.write(resp.content)
            img = Image.open(f"frame_{i}.jpg").convert("RGB").resize((VIDEO_W, VIDEO_H))
            break
        except Exception as e:
            print(f"Segment {i}: image attempt {attempt+1} failed ({e}), retrying...")
            time.sleep(3)

    if img is None:
        print(f"Segment {i}: falling back to a plain background.")
        img = Image.new("RGB", (VIDEO_W, VIDEO_H), (15, 15, 20))

    draw = ImageDraw.Draw(img)
    lines = wrap_text(seg["text"], font, VIDEO_W - 100, draw)
    line_height = 75
    total_h = line_height * len(lines)
    y = VIDEO_H - total_h - 220
    for line in lines:
        w = draw.textlength(line, font=font)
        x = (VIDEO_W - w) / 2
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
            draw.text((x + dx, y + dy), line, font=font, fill="black")
        draw.text((x, y), line, font=font, fill="white")
        y += line_height

    frame_path = f"frame_{i}_captioned.jpg"
    img.save(frame_path)
    frame_paths.append(frame_path)
    print(f"Segment {i} ready ({seg['duration']:.1f}s): {seg['text'][:60]}...")

clips = [ImageClip(p).set_duration(seg["duration"]) for p, seg in zip(frame_paths, segments)]
video = concatenate_videoclips(clips, method="compose")
video = video.set_audio(audio_clip)
video.write_videofile("shorts_video.mp4", fps=24, codec="libx264", audio_codec="aac")
print("\nVideo ready: shorts_video.mp4")

# ---------------------------------------------------------------------
# 7. Upload to YouTube (reuses saved token — no interactive login here)
# ---------------------------------------------------------------------
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
            "Re-run the one-time manual login step in Colab to generate a fresh youtube_token.json, "
            "then update the YOUTUBE_TOKEN_JSON secret in GitHub."
        )

youtube = build("youtube", "v3", credentials=credentials)

seo_output = getattr(seo_task.output, "raw", str(seo_task.output))


def extract_seo_title(text):
    match = re.search(r"1\.\s*[\"\u201c]?(.+?)[\"\u201d]?\s*(?:\(|$)", text)
    return match.group(1).strip() if match else "Automated Video"


video_title = extract_seo_title(seo_output)[:100]
video_description = seo_output[:4900]

request_body = {
    "snippet": {
        "title": video_title,
        "description": video_description,
        "tags": ["true crime", "psychology facts", "dark psychology"],
        "categoryId": "24",
    },
    "status": {
        "privacyStatus": "private",  # keep this until you've watched a few uploads and trust it
        "selfDeclaredMadeForKids": False,
    },
}

media = MediaFileUpload("shorts_video.mp4", chunksize=-1, resumable=True)

upload_request = youtube.videos().insert(part="snippet,status", body=request_body, media_body=media)

print("Uploading...")
response = upload_request.execute()
print(f"\nUploaded! Video ID: {response['id']}")
print(f"View it (private, only you can see it) at: https://youtu.be/{response['id']}")
