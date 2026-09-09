"""
Motive Unknown — Automated Narrator-Led 2D Animated History Pipeline.
Runs unattended on GitHub Actions.
"""

import os
import re
import time
import json
import asyncio
import numpy as np
from typing import List, Literal, Optional
from pydantic import BaseModel
from urllib.parse import quote
from PIL import Image, ImageDraw, ImageFont

import requests as req
import edge_tts
from scipy.io import wavfile
import pydub

# ---------------------------------------------------------------------
# 0. Secrets & Auth Setup
# ---------------------------------------------------------------------
NVIDIA_KEY = os.environ["NVIDIA_NIM_API_KEY"]

with open("youtube_token.json", "w") as f:
    f.write(os.environ["YOUTUBE_TOKEN_JSON"])
with open("client_secret.json", "w") as f:
    f.write(os.environ["YOUTUBE_CLIENT_SECRET_JSON"])

print("Secrets loaded.")

# ---------------------------------------------------------------------
# 1. LLM & Agents Initialization
# ---------------------------------------------------------------------
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import tool
from duckduckgo_search import DDGS
# Updated LLM config with explicit strict timeouts and retries
llm = LLM(
    model="openai/nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=NVIDIA_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    timeout=120,          # Force request timeout after 2 minutes so it doesn't hang forever
    max_retries=3,        # Retry automatically if connection drops
    temperature=0.7,
)
@tool("Web Search")
def search_tool(query: str) -> str:
    """Searches the web using DuckDuckGo."""
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(keywords=query, max_results=6))
            if not results:
                return "No results found."
            return "\n\n".join([f"{r.get('title','')}\n{r.get('body','')}\n{r.get('href','')}" for r in results])
        except Exception:
            time.sleep(2 * (attempt + 1))
    return "Search failed."

topic_scout = Agent(
    role="Topic Scout",
    goal="Find weird, dramatic, or hilarious historical events ideal for animated storytelling.",
    backstory="You find absurd historical stories like the War of the Bucket, Emu War, or bizarre ancient laws.",
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

researcher = Agent(
    role="Video Researcher",
    goal="Gather chronological plot points, funny historical details, and quote references for the story.",
    backstory="You dig up exact timelines, absurd quotes, and strange facts that drive narrative history channels.",
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

scriptwriter = Agent(
    role="Narrative Scriptwriter",
    goal="Write an engaging historical story driven by a Narrator with brief comedic dialogue scenes.",
    backstory="You write fast-paced animated history scripts matching channel formats like OverSimplified.",
    llm=llm,
    verbose=True,
    max_iter=3,  # Prevent infinite loops if LLM fails structured parsing
)
seo_specialist = Agent(
    role="YouTube SEO Specialist",
    goal="Generate high-CTR history channel titles, descriptions, tags, and thumbnail prompts.",
    backstory="You optimize videos for viral history animation audiences.",
    llm=llm,
    verbose=True,
)

# ---------------------------------------------------------------------
# 2. Pydantic Script Schema
# ---------------------------------------------------------------------
class ScriptSegment(BaseModel):
    speaker: Literal["NARRATOR", "CHARACTER_A", "CHARACTER_B"]
    text: str
    display_mode: Literal["narration_focus", "character_dialogue"]
    expression_A: Optional[Literal["eyes_neutral", "eyes_happy", "eyes_angry", "eyes_worried"]] = "eyes_neutral"
    expression_B: Optional[Literal["eyes_neutral", "eyes_happy", "eyes_angry", "eyes_worried"]] = "eyes_neutral"
    outfit_A: str
    outfit_B: str

class AnimatedStoryScript(BaseModel):
    segments: List[ScriptSegment]

topic_task = Task(
    description="Find one engaging, bizarre, or dramatic history story perfect for an animated video.",
    expected_output="Story topic and justification.",
    agent=topic_scout,
)

research_task = Task(
    description="Find the step-by-step narrative beats and funny details of the chosen topic.",
    expected_output="Chronological list of story events and character moments.",
    agent=researcher,
    context=[topic_task],
)

script_task = Task(
    description=(
        "Write a short, engaging animated history script.\n"
        "- Generate 12 to 16 scene segments in total.\n"
        "- Use 'NARRATOR' for general story narration.\n"
        "- Use 'CHARACTER_A' and 'CHARACTER_B' for funny character interactions.\n"
        "- Ensure 'display_mode' is set to 'narration_focus' for Narrator lines, and 'character_dialogue' for character lines."
    ),
    expected_output="Valid JSON matching AnimatedStoryScript schema.",
    agent=scriptwriter,
    context=[research_task],
    output_pydantic=AnimatedStoryScript,
)

seo_task = Task(
    description="Generate title, description, tags, and thumbnail prompt.",
    expected_output="SEO Metadata.",
    agent=seo_specialist,
    context=[script_task],
)

crew = Crew(
    agents=[topic_scout, researcher, scriptwriter, seo_specialist],
    tasks=[topic_task, research_task, script_task, seo_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
script_data = script_task.output.pydantic.model_dump()["segments"]

# ---------------------------------------------------------------------
# 3. Audio Engine (Narrator + Character Voices)
# ---------------------------------------------------------------------
VOICE_NARRATOR = "en-US-AndrewNeural"     # Energetic storytelling voice
VOICE_CHAR_A   = "en-US-GuyNeural"        # Character A voice
VOICE_CHAR_B   = "en-US-ChristopherNeural"# Character B voice

async def generate_script_audio(segments):
    combined = pydub.AudioSegment.empty()
    audio_info = []
    
    for idx, seg in enumerate(segments):
        speaker = seg["speaker"]
        if speaker == "NARRATOR":
            voice = VOICE_NARRATOR
        elif speaker == "CHARACTER_A":
            voice = VOICE_CHAR_A
        else:
            voice = VOICE_CHAR_B
            
        fn_mp3 = f"line_{idx}.mp3"
        fn_wav = f"line_{idx}.wav"
        
        comm = edge_tts.Communicate(text=seg["text"], voice=voice)
        await comm.save(fn_mp3)
        
        audio = pydub.AudioSegment.from_mp3(fn_mp3)
        audio.export(fn_wav, format="wav")
        
        audio_info.append({"file": fn_wav, "duration": audio.duration_seconds})
        combined += audio
        
    combined.export("full_audio.mp3", format="mp3")
    return audio_info

audio_segments = asyncio.run(generate_script_audio(script_data))

# ---------------------------------------------------------------------
# 4. Asset Renderer & Compositor
# ---------------------------------------------------------------------
CANVAS_W, CANVAS_H = 800, 1000

ANCHORS = {
    "eyes": (310, 260),
    "mouth": (350, 360),
    "hair": (260, 120),
    "beard": (300, 330),
    "outfit": (150, 420),
}

def load_png(filename):
    if filename and not filename.endswith(".png"):
        filename += ".png"
    if filename and os.path.exists(filename):
        return Image.open(filename).convert("RGBA")
    return None

def draw_eyebrows(canvas, expression):
    draw = ImageDraw.Draw(canvas)
    lx1, ly1, lx2, ly2 = 330, 245, 375, 245
    rx1, ry1, rx2, ry2 = 425, 245, 470, 245
    
    if expression == "eyes_angry":
        ly2 += 18
        ry1 += 18
    elif expression == "eyes_worried":
        ly1 += 18
        ry2 += 18
    elif expression == "eyes_happy":
        ly1 -= 10
        ly2 -= 10
        ry1 -= 10
        ry2 -= 10

    line_color = (35, 25, 20, 255)
    draw.line([(lx1, ly1), (lx2, ly2)], fill=line_color, width=9)
    draw.line([(rx1, ry1), (rx2, ry2)], fill=line_color, width=9)
    return canvas

def compose_character(expression, hair, beard, outfit, mouth_state, is_flipped=False):
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    
    base = load_png("base_body_1.png")
    if base:
        if base.size != (CANVAS_W, CANVAS_H):
            base = base.resize((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)
        canvas.alpha_composite(base)
        
    eyes_img = load_png(f"{expression}.png")
    if eyes_img: canvas.alpha_composite(eyes_img, ANCHORS["eyes"])
    
    mouth_img = load_png(f"{mouth_state}.png")
    if mouth_img: canvas.alpha_composite(mouth_img, ANCHORS["mouth"])
    
    hair_img = load_png(f"{hair}.png")
    if hair_img: canvas.alpha_composite(hair_img, ANCHORS["hair"])
    
    beard_img = load_png(f"{beard}.png")
    if beard_img: canvas.alpha_composite(beard_img, ANCHORS["beard"])
    
    outfit_img = load_png(f"{outfit}.png")
    if outfit_img: canvas.alpha_composite(outfit_img, ANCHORS["outfit"])
    
    canvas = draw_eyebrows(canvas, expression)
    
    if is_flipped:
        canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
        
    return canvas

# ---------------------------------------------------------------------
# 5. Video Stage & Animation Loop
# ---------------------------------------------------------------------
def generate_video(script, audio_info):
    from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
    
    movie_clips = []
    VIDEO_W, VIDEO_H = 1920, 1080
    
    for idx, (seg, audio) in enumerate(zip(script, audio_info)):
        sample_rate, data = wavfile.read(audio["file"])
        if len(data.shape) > 1:
            data = data.mean(axis=1)
            
        fps = 12
        total_frames = int(audio["duration"] * fps)
        samples_per_frame = int(len(data) / max(total_frames, 1))
        
        frame_clips = []
        speaker = seg["speaker"]
        
        for f in range(total_frames):
            start_i = f * samples_per_frame
            end_i = min((f + 1) * samples_per_frame, len(data))
            chunk = data[start_i:end_i]
            amplitude = np.max(np.abs(chunk)) if len(chunk) > 0 else 0
            
            if amplitude > 10000:
                mouth = "mouth_fully_open"
            elif amplitude > 3000:
                mouth = "mouth_half_open"
            else:
                mouth = "mouth_closed"
                
            mouth_a = mouth if speaker == "CHARACTER_A" else "mouth_closed"
            mouth_b = mouth if speaker == "CHARACTER_B" else "mouth_closed"
            
            bg = Image.new("RGBA", (VIDEO_W, VIDEO_H), (240, 240, 245, 255))
            
            if seg["display_mode"] == "narration_focus":
                # Center a single character during narrator explanations
                char_center = compose_character(
                    expression=seg.get("expression_A", "eyes_neutral"),
                    hair="hair_short",
                    beard="no_beard",
                    outfit=seg.get("outfit_A", "rome_commoner_tunic"),
                    mouth_state="mouth_closed",
                    is_flipped=False
                )
                bg.alpha_composite(char_center, (560, 80))
            else:
                # Stage two characters for comedy dialogue scenes
                char_a = compose_character(
                    expression=seg.get("expression_A", "eyes_neutral"),
                    hair="hair_short",
                    beard="no_beard",
                    outfit=seg.get("outfit_A", "rome_commoner_tunic"),
                    mouth_state=mouth_a,
                    is_flipped=False
                )
                char_b = compose_character(
                    expression=seg.get("expression_B", "eyes_neutral"),
                    hair="hair_long",
                    beard="beard",
                    outfit=seg.get("outfit_B", "rome_soldier_armor"),
                    mouth_state=mouth_b,
                    is_flipped=True
                )
                bg.alpha_composite(char_a, (100, 80))
                bg.alpha_composite(char_b, (1020, 80))
                
            frame_file = f"frame_{idx}_{f}.png"
            bg.convert("RGB").save(frame_file)
            frame_clips.append(ImageClip(frame_file).set_duration(1.0 / fps))
            
        line_video = concatenate_videoclips(frame_clips, method="compose")
        line_video = line_video.set_audio(AudioFileClip(audio["file"]))
        movie_clips.append(line_video)
        
    final_video = concatenate_videoclips(movie_clips, method="compose")
    final_video.write_videofile("final_video.mp4", fps=12, codec="libx264", audio_codec="aac")

generate_video(script_data, audio_segments)

# ---------------------------------------------------------------------
# 6. YouTube Upload & Thumbnail
# ---------------------------------------------------------------------
seo_output = getattr(seo_task.output, "raw", str(seo_task.output))

thumb_prompt = "2D cartoon history animation, funny historical moment, high contrast"
thumb_url = f"https://image.pollinations.ai/prompt/{quote(thumb_prompt)}?width=1280&height=720&nologo=true"

try:
    r = req.get(thumb_url, timeout=30)
    if r.status_code == 200:
        with open("thumbnail.jpg", "wb") as f:
            f.write(r.content)
except Exception:
    pass

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]
credentials = Credentials.from_authorized_user_file("youtube_token.json", SCOPES)

if credentials.expired and credentials.refresh_token:
    credentials.refresh(Request())

youtube = build("youtube", "v3", credentials=credentials)

request_body = {
    "snippet": {
        "title": "Historical Events That Make No Sense",
        "description": seo_output[:4500],
        "tags": ["history", "animation", "oversimplified", "education"],
        "categoryId": "23",
    },
    "status": {
        "privacyStatus": "private",
        "selfDeclaredMadeForKids": False,
    },
}

media = MediaFileUpload("final_video.mp4", chunksize=-1, resumable=True)
upload_request = youtube.videos().insert(part="snippet,status", body=request_body, media_body=media)

response = upload_request.execute()
video_id = response["id"]
print(f"Uploaded! View privately at: https://youtu.be/{video_id}")

if os.path.exists("thumbnail.jpg"):
    try:
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload("thumbnail.jpg")
        ).execute()
        print("Thumbnail applied.")
    except Exception as e:
        print(f"Thumbnail skipped: {e}")
