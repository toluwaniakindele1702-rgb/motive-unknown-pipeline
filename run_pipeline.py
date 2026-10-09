"""Relic Loop production wrapper: Shorts are sequential cuts of the finished long video."""
from __future__ import annotations
import argparse
from pathlib import Path
import pipeline as p
import runtime_script_repair
import gemini_hardening
import production_hardening


# Permanent Relic Loop production policy.
# 5–7 minutes is the target runtime; do not pad a complete story just to hit a word count.
p.SCRIPT_MIN_WORDS = 750
p.SCRIPT_MAX_WORDS = 1150
# More frequent meaningful visual changes: allow up to 150 visual beats per episode.
p.MAX_VISUAL_BEATS_PER_VIDEO = 150
# The original pipeline repair routine targeted ~2,050 words. Replace it with
# the runtime-aware 750–1,150-word repair before every production run.
runtime_script_repair.install(p)


# Harden every generated still prompt around the canonical RL character/style.
_original_make_visual_prompt = p.make_visual_prompt


def _rl_hardened_visual_prompt(*args, **kwargs):
    prompt = _original_make_visual_prompt(*args, **kwargs)
    rl_rules = """
RELIC LOOP CHARACTER / STYLE LOCK — FOLLOW STRICTLY:
- RL is the canonical recurring human protagonist: a young adult Black male with dark hair, friendly curious face, blue hoodie/jacket over a cream shirt, dark trousers/cargo pants, sneakers, and a backpack when appropriate.
- Preserve RL's face, hairstyle, skin tone, clothing palette, body proportions, and overall illustrated identity across scenes. Do not redesign or substitute him with a generic male character.
- Use the established Relic Loop clean modern 2D educational/explainer illustration style: polished, sharp, readable, expressive, simple shapes, strong composition, and clear visual storytelling.
- Do NOT turn RL or the scene into photorealism, 3D/Pixar style, anime, a different cartoon style, or an old-fashioned documentary illustration.
- RL should appear naturally in the scene only when a human/character is useful; object-only diagrams, animals, environments, mechanisms, maps, and historical/real-person scenes may omit RL when the narration requires it.
- When RL is present, keep him visually consistent rather than inventing a new outfit, face, hairstyle, age, or character design.
- The image must directly explain the narrated beat. Do not add unrelated decorative imagery just to fill the frame.
"""
    return prompt + "\n\n" + rl_rules.strip()


p.make_visual_prompt = _rl_hardened_visual_prompt


def _scene_ranges(script):
    starts=[]; durations=[]; cursor=0.0
    for i in range(1, len(script["scenes"])+1):
        d=p.audio_duration(p.AUDIO_DIR / f"scene_{i:03d}.wav")
        starts.append(cursor); durations.append(d); cursor += d
    return starts, durations, cursor


def _choose_ranges(script):
    starts, durations, total = _scene_ranges(script)
    if total <= 0 or total < 60: return []
    count = min(5, max(3, int((total + 179.999)//180)))
    while count < 5 and total / count > 179.5:
        count += 1
    targets = [total * i / count for i in range(1, count)]
    boundaries=[]; used_scene=0
    for target in targets:
        best=None; best_err=float('inf')
        for idx in range(used_scene+1, len(durations)+1):
            end = starts[idx-1] + durations[idx-1]
            err=abs(end-target)
            if err < best_err:
                best_err=err; best=idx
        if best is None or best <= used_scene or best >= len(durations):
            continue
        boundaries.append(best); used_scene=best
    boundaries=sorted(set(boundaries))
    ranges=[]; start_scene=1
    for end_scene in boundaries+[len(durations)]:
        if end_scene < start_scene: continue
        start=starts[start_scene-1]
        end=starts[end_scene-1]+durations[end_scene-1]
        dur=end-start
        if dur > 180.0: return []
        ranges.append((start_scene,end_scene,start,dur)); start_scene=end_scene+1
    return ranges if len(ranges) >= 3 else []


def _render_segment(video_path, start, duration, out_path):
    # Direct cut/reframe of the finished long-form video. No subtitles, no new narration,
    # no new visuals, and no AI Short-specific generation.
    vf=("scale=1080:608:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:0:656:color=black,setsar=1")
    p.run_cmd(["ffmpeg","-y","-ss",f"{start:.3f}","-i",str(video_path),"-t",f"{duration:.3f}",
               "-vf",vf,"-r",str(p.VIDEO_FPS),"-c:v","libx264","-preset","veryfast",
               "-crf","21","-c:a","aac","-b:a","160k","-movflags","+faststart",str(out_path)],
              f"render sequential Short {out_path.stem}")


def sequential_shorts(video_path, script, seo, long_video_id):
    if not p.SHORTS_ENABLED:
        print("[SHORTS] disabled")
        return []
    ranges=_choose_ranges(script)
    if not ranges:
        print("[SHORTS] Could not make 3-5 duration-safe sequential parts; skipping Shorts.")
        return []
    uploaded=[]
    manifest={"shorts":{}}
    for n,(start_scene,end_scene,start,duration) in enumerate(ranges,1):
        key=f"{long_video_id}-part-{n}-{start_scene}-{end_scene}"
        path=p.OUTPUT_DIR/f"short_{n:02d}.mp4"
        if not path.exists() or path.stat().st_size<10000:
            _render_segment(video_path,start,duration,path)
        title=f"{seo['title']} — Part {n}/{len(ranges)}"[:100]
        desc=f"Part {n} of {len(ranges)} from the full Relic Loop episode.\n\nFull video: https://youtu.be/{long_video_id}"
        try:
            vid=p._upload_short(path,title,desc,seo.get("tags",[]),key)
            uploaded.append(vid)
            manifest["shorts"][key]={"video_id":vid,"part":n,"parts_total":len(ranges),"start_scene":start_scene,"end_scene":end_scene,"start_seconds":round(start,2),"duration_seconds":round(duration,2)}
            p._save_current_json(p.CURRENT_SHORTS_PATH,manifest)
            print(f"[SHORTS] PART {n}/{len(ranges)}: {vid} ({duration:.1f}s) scenes {start_scene}-{end_scene}")
        except Exception as exc:
            print(f"[SHORTS] upload failed for Part {n}: {exc}")
    p.checkpoint("shorts_complete",count=len(uploaded),expected=len(ranges),long_video_id=long_video_id,mode="sequential_cuts_no_subtitles")
    return uploaded


def main(mode):
    # Patch before production. Shorts consume only the completed long-form file.
    p.generate_and_upload_shorts=sequential_shorts
    p.main(mode)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",default="full",choices=["full","voice_test","visual_test","local_image_test"])
    args=ap.parse_args()
    main(args.mode)
