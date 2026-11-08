#!/usr/bin/env python3
"""Chispa-Pifia V2: AI writes the story; Python owns every structural invariant."""
from __future__ import annotations
import argparse, json, os, re, time
from datetime import datetime, timezone
from pathlib import Path
from groq import Groq

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"state"/"current_run"
MODEL=os.getenv("GROQ_WRITER_MODEL","openai/gpt-oss-120b")
REVIEW_MODEL=os.getenv("GROQ_REVIEWER_MODEL","openai/gpt-oss-20b")
KEY=os.getenv("GROQ_API_KEY","").strip()
SCENES=11

STORY_SYSTEM="""You are the head comedy writer for CHISPA-PIFIA, a short-form channel built around absurd, coherent, visual stickman comedy.

WRITE A TINY COMEDY FILM, NOT RANDOM AI GAGS.
Build ONE original story around one clear human goal and one unusual but understandable obstacle, object, misunderstanding, rule or bad idea. The protagonist must actively choose what to do. Their first solution is understandable but flawed; that choice causes the next problem. They then double down or improvise, making things worse. Include at least three distinct escalation steps, each caused by the previous choice or consequence. End with a specific, satisfying punchline that pays off something planted earlier. A twist is optional; never force one.

QUALITY RULES:
- First sentence starts inside the funniest, strangest or most urgent moment. No greeting, biography, "one day", "have you ever", or channel intro.
- Give the protagonist a concrete goal the audience understands immediately.
- Use a fresh, specific premise. Avoid generic laziness, random aliens, unexplained magic, generic school embarrassment and familiar viral-story clichés unless the execution is genuinely original.
- Events must connect through simple cause and effect. Escalation should become increasingly ridiculous without becoming confusing.
- Use concrete physical actions, visible objects and simple settings that can be illustrated in 2D.
- Protagonist has agency; events do not merely happen to them.
- Final line is the punchline. No lesson, moral, summary, subscribe request or generic outro.
- Natural conversational English, punchy spoken rhythm, varied sentence openings, no filler.
- No screenplay directions, dialogue formatting, scene labels, camera directions or image prompts.
- Safe, non-graphic, non-sexual comedy. No dangerous stunts or harmful instructions.

LENGTH: 125–145 spoken words; aim for 130–138. Count narration only.

OUTPUT: Return exactly one valid JSON object, no markdown, with ONLY:
{"title":"short curiosity-building title","narration":"complete spoken story","protagonist":"short character name or role","supporting_characters":["zero to two names or roles"],"setting":"main location in a few words"}
Do not create scenes, beats, excerpts, visual prompts, story outlines, metadata or extra keys. Python creates those."""

REVIEW_SYSTEM="""Act as a strict comedy script editor. Score this story 1–10 for immediate hook, originality, clarity on first listen, protagonist agency, causal escalation, visual filmability and punchline. Penalize random events, passive protagonist, repetitive escalation, generic AI phrasing and endings that explain the joke. Do not require a twist. Return JSON only: {"score":1,"main_problem":"one sentence","rewrite_instruction":"specific actionable fix"}."""
REWRITE_SYSTEM="""Rewrite the supplied Chispa-Pifia narration to fix the editor's criticism. Preserve its core premise only if it is worth keeping. Ensure an immediate hook, active protagonist, causal escalation and a satisfying final punchline. Natural spoken English, concrete visual actions, 125–145 narration words (ideally 130–138). Return exactly the compact JSON keys title,narration,protagonist,supporting_characters,setting. No scenes, prompts, metadata or commentary."""
VISUAL_SYSTEM="""You are a top-tier storyboard artist for polished 2D stickman comedy. For EACH numbered narration excerpt, design one distinct frozen image literally depicting that exact moment. Excerpt is source of truth: do not invent plot events, props, characters or locations. Preserve character identity, outfit, prop state, screen direction and setting continuity. Make each frame readable on a phone.

Match approved Chispa-Pifia reference images exactly: same expressive 2D stickman, pale round head, eyes/eyebrows/mouth, proportions and outline weight. All environments and props are also illustrated 2D cartoons, never photographic or 3D. Vertical 9:16, strong silhouette, expressive body language, one dominant action, clear depth and readable lighting. Vary shot sizes and angles when useful.

Scene 1 is the opening hook: close or medium shot of the protagonist actively confronting the opening problem; key object/consequence large and obvious. Never neutral standing pose or empty establishing shot.

ZERO TEXT: no captions, words, letters, numbers, symbols, speech bubbles, signs, logos, watermarks, interface text or typography anywhere.

Return JSON only:
{"scenes":[{"number":1,"visual_prompt":"40–75 words for one specific frame","camera":"shot/composition direction","action":"specific visible action","mood":"one or two words"}]}
Return exactly the requested scene numbers. Do not return or rewrite narration."""

def norm(s): return " ".join(str(s or "").split())
def spans(s): return list(re.finditer(r"\b[\w'’\-]+\b",s))
def wc(s): return len(spans(s))
def parse_json(s):
    s=(s or "").strip().replace(chr(96)*3+"json","").replace(chr(96)*3,"").strip()
    # Some Groq model responses include a short preamble despite JSON-only prompting.
    # Extract the outermost object, then let Python validate the JSON strictly.
    start=s.find("{")
    end=s.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object")
    x=json.loads(s[start:end+1])
    if not isinstance(x,dict): raise ValueError("Expected JSON object")
    return x
def call(client,model,system,user,tokens=1500,temp=0.65):
    # Do not request Groq's server-side json_object constrained decoding here.
    # It returned HTTP 400 json_validate_failed for openai/gpt-oss models before
    # generation began. The prompts already require JSON; parse_json extracts and
    # validates the returned object locally, allowing retries to handle bad output.
    r=client.chat.completions.create(model=model,messages=[{"role":"system","content":system},{"role":"user","content":user}],
      temperature=temp,max_completion_tokens=tokens)
    return parse_json(r.choices[0].message.content or "")
def validate_story(raw):
    s={"title":norm(raw.get("title"))[:90],"narration":norm(raw.get("narration")),
       "protagonist":norm(raw.get("protagonist")) or "The protagonist",
       "setting":norm(raw.get("setting")) or "An everyday location",
       "supporting_characters":raw.get("supporting_characters",[])}
    if not s["title"]: raise ValueError("Missing title")
    if not 125<=wc(s["narration"])<=145: raise ValueError(f"Narration has {wc(s['narration'])} words; required 125–145")
    if not isinstance(s["supporting_characters"],list): s["supporting_characters"]=[]
    s["supporting_characters"]=[norm(x) for x in s["supporting_characters"] if norm(x)][:2]
    return s
def split_exact(text,count=SCENES):
    text=norm(text); w=spans(text); total=len(w)
    if total<count*8: raise ValueError("Not enough narration words for scene split")
    cuts=[]; prev=0
    for i in range(1,count):
        ideal=round(i*total/count); lo=prev+8; hi=total-(count-i)*8
        def cost(pos):
            gap=text[w[pos-1].end():w[pos].start()]
            penalty=0 if re.search(r"[.!?][\"')\]]?$",gap) else 0.5 if re.search(r"[;:][\"')\]]?$",gap) else 1 if "," in gap else 2
            return (abs(pos-ideal)+penalty,abs(pos-ideal))
        chosen=min(range(lo,hi+1),key=cost); cuts.append(w[chosen].start()); prev=chosen
    out=[]; start=0
    for cut in cuts: out.append(text[start:cut].strip()); start=cut
    out.append(text[start:].strip())
    if len(out)!=count or " ".join(out)!=text: raise ValueError("Exact scene split invariant failed")
    return out
def fallback_visual(scene,first=False):
    camera=("Close-medium vertical 9:16 shot, slightly low angle, protagonist and key object large in frame."
      if first else "Medium vertical 9:16 shot, clear foreground action and enough setting to preserve continuity.")
    prompt=("One polished vertical 9:16 illustrated 2D cartoon frame in the approved Chispa-Pifia stickman style. "
      "Show the same expressive pale round-headed stickman protagonist, matching approved face, proportions, outfit and bold outlines. "
      "Literally depict this exact moment: "+scene["narration"]+
      ". Show the specific physical action, facial expression and relevant story object. Use a coherent hand-drawn cartoon environment, "
      "clear silhouette, one dominant focal action, readable lighting and foreground/background separation. Maintain character and prop continuity. "
      "No text, letters, symbols, captions or watermark.")
    return {"visual_prompt":prompt,"camera":camera,"action":"Literally reenact the supplied narration excerpt with the protagonist visibly acting.","mood":"expressive comedy"}
def visual_prompts(client,title,narration,characters,setting,scenes):
    for offset in range(0,len(scenes),3):
        batch=scenes[offset:offset+3]; accepted={}
        payload={"title":title,"setting":setting,"characters":[{"name":c["name"],"role":c["role"]} for c in characters],
          "full_story_for_continuity":narration,"requested_scenes":[{"number":s["number"],"exact_excerpt":s["narration"]} for s in batch]}
        for attempt in range(2):
            try:
                obj=call(client,MODEL,VISUAL_SYSTEM,json.dumps(payload,ensure_ascii=False),1500,0.5)
                returned=obj.get("scenes")
                if not isinstance(returned,list): raise ValueError("Missing scenes array")
                bynum={}
                for x in returned:
                    if isinstance(x,dict):
                        try: bynum[int(x.get("number"))]=x
                        except (TypeError,ValueError): pass
                candidate={}
                for s in batch:
                    x=bynum.get(s["number"],{}); p=norm(x.get("visual_prompt")); cam=norm(x.get("camera")); act=norm(x.get("action"))
                    if "stickman" not in p.lower() or not 30<=wc(p)<=100 or len(cam)<8 or len(act)<10:
                        raise ValueError(f"Scene {s['number']} incomplete (visual words={wc(p)})")
                    candidate[s["number"]]={"visual_prompt":p,"camera":cam,"action":act,"mood":norm(x.get("mood")) or "comedy"}
                accepted=candidate; break
            except Exception as e:
                print(f"[V2 VISUAL] batch {offset//3+1} attempt {attempt+1}/2: {type(e).__name__}: {e}")
                if attempt==0: time.sleep(2)
        for s in batch:
            s.update(accepted.get(s["number"]) or fallback_visual(s,s["number"]==1))
            if s["number"] not in accepted: print(f"[V2 VISUAL] scene {s['number']}: deterministic fallback used")
def review_story(client,story):
    review={"score":None,"main_problem":"Review unavailable","rewrite_attempted":False}
    try:
        review=call(client,REVIEW_MODEL,REVIEW_SYSTEM,json.dumps({"title":story["title"],"narration":story["narration"]}),300,0.2)
        score=int(review.get("score",0)); print(f"[V2 REVIEW] {score}/10: {norm(review.get('main_problem'))}")
        if score<7 and norm(review.get("rewrite_instruction")):
            revised=call(client,MODEL,REWRITE_SYSTEM,json.dumps({"current_story":story,"editor_instruction":norm(review["rewrite_instruction"])}),1500,0.5)
            try: story=validate_story(revised); review["rewrite_attempted"]=True; print("[V2 REVIEW] one editorial rewrite accepted")
            except Exception as e: print(f"[V2 REVIEW] rewrite rejected; retaining original: {e}")
    except Exception as e: print(f"[V2 REVIEW] advisory review skipped: {type(e).__name__}: {e}")
    return story,review
def self_test():
    sample=norm("The shop owner offered Max one free balloon if he carried it home carefully. Max tied it to his wrist, then decided the safest place was inside his jacket. The balloon pushed the jacket open, so he added a belt. The belt trapped his arms. He tried walking sideways, but the balloon kept turning him toward the shop. A gust pulled him backwards, and he grabbed a street sign to stop. The owner ran over and cut the string. Max thanked him for saving the balloon. Then the owner pointed at the tiny receipt still tied to Max's wrist.")
    parts=split_exact(sample,11)
    assert len(parts)==11 and " ".join(parts)==sample and sum(wc(x) for x in parts)==wc(sample)
    print(f"[V2 SELF-TEST] PASS: {len(parts)} exact excerpts, {wc(sample)} words")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); args=ap.parse_args()
    if args.self_test: self_test(); return
    if not KEY: raise SystemExit("GROQ_API_KEY is missing")
    RUN.mkdir(parents=True,exist_ok=True); client=Groq(api_key=KEY)
    user="Write one fresh original Chispa-Pifia story. 125–145 spoken words, ideally 130–138. Immediate visual hook, protagonist with a clear goal, three causal escalations, and a satisfying final punchline."
    topic=os.getenv("RELIC_LOOP_TOPIC","").strip()
    if topic: user+=" Optional story seed (use only if it improves the story): "+topic
    history_path=ROOT/"state"/"story_history.json"
    try:
        old=json.loads(history_path.read_text()); titles=[norm(x) for x in old if norm(x)]
        if titles: user+=" Avoid repeating these recent titles: "+"; ".join(titles[-12:])
    except Exception: pass
    raw=None; last=None
    for attempt in range(2):
        try: raw=call(client,MODEL,STORY_SYSTEM,user,1700,0.78 if attempt==0 else 0.58); validate_story(raw); break
        except Exception as e:
            last=e; print(f"[V2 STORY] attempt {attempt+1}/2: {type(e).__name__}: {e}")
            if attempt==0: time.sleep(3)
    if raw is None: raise SystemExit(f"Compact story generation failed: {last}")
    story,review=review_story(client,validate_story(raw))
    excerpts=split_exact(story["narration"],SCENES)
    characters=[{"name":story["protagonist"],"role":"human stickman protagonist",
      "appearance":"The approved canonical Chispa-Pifia 2D stickman design; preserve round pale head, facial design, proportions, outline weight and established outfit."}]
    for name in story["supporting_characters"]:
        characters.append({"name":name,"role":"human stickman supporting character","appearance":"Supporting stickman in the same approved 2D illustrated world, stylistically consistent."})
    scenes=[]; total=wc(story["narration"])
    for i,excerpt in enumerate(excerpts,1):
        scenes.append({"number":i,"duration_seconds":round(65*wc(excerpt)/total,2),"beat_text":excerpt,"narration":excerpt,
          "visual_prompt":"","camera":"","action":"","mood":""})
    visual_prompts(client,story["title"],story["narration"],characters,story["setting"],scenes)
    package={"title":story["title"],"hook":scenes[0]["narration"],"narration":story["narration"],"characters":characters,
      "locations":[story["setting"]],"story_structure":{"problem":scenes[0]["narration"],"goal":f"{story['protagonist']} has a clear goal in {story['setting']}.",
      "causal_chain":[s["narration"] for s in scenes],"escalations":[scenes[i]["narration"] for i in (2,5,8)],
      "twist":"No forced twist; the final beat pays off the central comic idea.","punchline":scenes[-1]["narration"]},
      "scenes":scenes,"flow_prompt":"Continuous absurd-but-coherent Chispa-Pifia 2D stickman comedy. Match canonical references. Illustrate each exact narration excerpt in order. Vertical 9:16. No text.",
      "youtube_description":f"{story['title']} — absurd stickman comedy from Chispa-Pifia. #shorts #stickman #comedy",
      "tags":["shorts","stickman","comedy","absurd comedy","storytime"],"story_quality_review":review,
      "generated_at":datetime.now(timezone.utc).isoformat(),"generation_mode":"chispa_pifia_v2_script_first_python_scene_split",
      "visual_policy":"Python owns scene count, exact narration excerpts, timing and reconstruction."}
    if " ".join(s["narration"] for s in scenes)!=package["narration"]: raise SystemExit("Exact narration reconstruction failed")
    (RUN/"relic_loop_short.json").write_text(json.dumps(package,ensure_ascii=False,indent=2),encoding="utf-8")
    (RUN/"flow_manifest.json").write_text(json.dumps({"title":package["title"],"description":package["youtube_description"],
      "scene_count":len(scenes),"narration_words":total,"generation_mode":package["generation_mode"]},indent=2),encoding="utf-8")
    try: old=json.loads(history_path.read_text())
    except Exception: old=[]
    old.append(package["title"]); history_path.parent.mkdir(parents=True,exist_ok=True)
    history_path.write_text(json.dumps(old[-100:],ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"[V2 STORY] READY: {package['title']} | {total} words | {len(scenes)} exact scenes")
if __name__=="__main__": main()
