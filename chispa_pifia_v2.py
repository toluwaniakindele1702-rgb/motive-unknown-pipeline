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

STORY_SYSTEM="""You are the head comedy writer for CHISPA-PIFIA, a short-form channel built around original, tightly structured absurd stickman comedy.

WRITE A COMPLETE COMEDY SHORT, NOT A RANDOM AI GAG.
Create ONE original story around a concrete human goal and one unusual but understandable obstacle, object, misunderstanding or bad decision. The protagonist must actively choose what to do. Their first solution should be understandable but flawed; that choice causes the next problem. They then improvise or double down, creating at least three distinct causal escalations. End with a specific, surprising-but-logical punchline that pays off an earlier detail.

QUALITY REQUIREMENTS:
- First sentence starts inside the funniest, strangest or most urgent moment. No greeting, biography, "one day", "have you ever", channel intro or generic setup.
- Establish the protagonist's goal and central comic problem within the first two sentences.
- Use a fresh, specific premise. Avoid generic laziness, random aliens, unexplained magic, school embarrassment, recycled viral-story setups and obvious AI clichés.
- Every escalation must be caused by the protagonist's previous action or a clear consequence. No unrelated gags.
- Use concrete physical actions, visible objects and simple locations that can be illustrated in 2D stickman scenes.
- Give the protagonist agency and distinct reactions. Supporting characters must have a clear function.
- Build toward one strongest final joke. The last sentence is the punchline; no moral, lesson, summary, subscribe request or generic outro.
- Natural conversational English, punchy spoken rhythm, varied sentence lengths and openings. Avoid filler and repetitive phrasing.
- No screenplay directions, scene labels, camera directions, image prompts or dialogue formatting.
- Safe, non-graphic, non-sexual comedy. No dangerous stunts or harmful instructions.

LENGTH: 180–205 spoken words. Count narration only. Aim for 188–198 words so the finished narration naturally fits a 50–90 second Short.

OUTPUT: Return exactly one valid JSON object, no markdown, with ONLY:
{"title":"short curiosity-building title","narration":"complete spoken story","protagonist":"short character name or role","supporting_characters":["zero to two names or roles"],"setting":"main location in a few words"}
Do not create scenes, beats, excerpts, visual prompts, outlines, metadata or extra keys. Python creates those."""

REVIEW_SYSTEM="""Act as a demanding professional comedy script editor. Score the story 1–10 using: immediate hook, originality, clarity on first listen, protagonist agency, causal escalation, visual filmability, rhythm and punchline payoff. A score of 7 means publishable with minor imperfections; 8 means strong; 9–10 is exceptional. Score 6 or below if the premise is generic, events are disconnected, the protagonist is passive, escalation repeats, the story is hard to visualize, the ending explains the joke, or the final punchline is weak. Do not inflate scores. Return JSON only: {"score":1,"main_problem":"one sentence","rewrite_instruction":"specific actionable fix"}."""
REWRITE_SYSTEM="""Rewrite the supplied Chispa-Pifia narration to fix the editor's criticism. Keep the premise only if it is original and strong. Make the opening immediate, protagonist active, escalations causally connected, and final punchline specific and earned. Use natural spoken English, concrete physical actions, varied sentence rhythm, and 180–205 narration words. Do not pad with filler. Return exactly the compact JSON keys title,narration,protagonist,supporting_characters,setting. No scenes, prompts, metadata or commentary."""
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
    choice=r.choices[0]
    content=choice.message.content or ""
    try:
        return parse_json(content)
    except Exception as exc:
        finish=getattr(choice,"finish_reason",None)
        usage=getattr(r,"usage",None)
        completion=getattr(usage,"completion_tokens",None) if usage else None
        print(f"[V2 MODEL] invalid JSON model={model} finish_reason={finish} completion_tokens={completion} content_preview={content[:500]!r}")
        raise ValueError(f"Model returned invalid/empty JSON (finish_reason={finish}, completion_tokens={completion}): {exc}") from exc
def validate_story(raw):
    s={"title":norm(raw.get("title"))[:90],"narration":norm(raw.get("narration")),
       "protagonist":norm(raw.get("protagonist")) or "The protagonist",
       "setting":norm(raw.get("setting")) or "An everyday location",
       "supporting_characters":raw.get("supporting_characters",[])}
    if not s["title"]: raise ValueError("Missing title")
    if not 180<=wc(s["narration"])<=205: raise ValueError(f"Narration has {wc(s['narration'])} words; required 180–205")
    if not isinstance(s["supporting_characters"],list): s["supporting_characters"]=[]
    s["supporting_characters"]=[norm(x) for x in s["supporting_characters"] if norm(x)][:2]
    return s
def split_exact(text,count=SCENES):
    """Split on sentence boundaries first; use clause/word boundaries only as a last resort."""
    text=norm(text); w=spans(text); total=len(w)
    minimum=8
    if total<count*minimum: raise ValueError("Not enough narration words for scene split")
    sentence=set(); clause=set()
    for pos in range(1,total):
        gap=text[w[pos-1].end():w[pos].start()].strip()
        if re.search(r"[.!?][\"')\]]*$",gap): sentence.add(pos)
        elif re.search(r"[;:—–,][\"')\]]*$",gap): clause.add(pos)
    ranked={p:0 for p in sentence}
    for p in clause: ranked.setdefault(p,8)
    for p in range(1,total): ranked.setdefault(p,24)
    cuts=[]; prev=0
    for i in range(1,count):
        ideal=round(i*total/count)
        lo=prev+minimum
        hi=total-(count-i)*minimum
        options=[p for p in ranked if lo<=p<=hi]
        if not options: raise ValueError("Could not create balanced sentence-aware scene split")
        chosen=min(options,key=lambda p:(abs(p-ideal)+ranked[p],abs(p-ideal),p))
        cuts.append(w[chosen].start()); prev=chosen
    out=[]; start=0
    for cut in cuts: out.append(text[start:cut].strip()); start=cut
    out.append(text[start:].strip())
    if len(out)!=count or " ".join(out)!=text: raise ValueError("Exact scene split invariant failed")
    if any(wc(x)<minimum for x in out): raise ValueError("Scene excerpt below minimum word count")
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
    for offset in range(0,len(scenes),2):
        batch=scenes[offset:offset+2]; accepted={}
        payload={"title":title,"setting":setting,"characters":[{"name":c["name"],"role":c["role"]} for c in characters],
          "full_story_for_continuity":narration,"requested_scenes":[{"number":s["number"],"exact_excerpt":s["narration"]} for s in batch]}
        for attempt in range(2):
            try:
                obj=call(client,MODEL,VISUAL_SYSTEM,json.dumps(payload,ensure_ascii=False),2800,0.5)
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
                print(f"[V2 VISUAL] batch {offset//2+1} attempt {attempt+1}/2: {type(e).__name__}: {e}")
                if attempt==0: time.sleep(2)
        for s in batch:
            s.update(accepted.get(s["number"]) or fallback_visual(s,s["number"]==1))
            if s["number"] not in accepted: print(f"[V2 VISUAL] scene {s['number']}: deterministic fallback used")
def review_story(client,story):
    """Require an editorial pass; rewrite once and re-score before acceptance."""
    review=call(client,REVIEW_MODEL,REVIEW_SYSTEM,
        json.dumps({"title":story["title"],"narration":story["narration"]}),900,0.2)
    score=int(review.get("score",0))
    print(f"[V2 REVIEW] initial score={score}/10: {norm(review.get('main_problem'))}")
    if score>=7:
        review["rewrite_attempted"]=False
        return story,review
    instruction=norm(review.get("rewrite_instruction"))
    if not instruction:
        raise ValueError("Story failed editorial gate and reviewer supplied no rewrite instruction")
    revised=validate_story(call(client,MODEL,REWRITE_SYSTEM,
        json.dumps({"current_story":story,"editor_instruction":instruction}),1800,0.55))
    second=call(client,REVIEW_MODEL,REVIEW_SYSTEM,
        json.dumps({"title":revised["title"],"narration":revised["narration"]}),900,0.2)
    second_score=int(second.get("score",0))
    print(f"[V2 REVIEW] rewritten score={second_score}/10: {norm(second.get('main_problem'))}")
    if second_score<7:
        raise ValueError(f"Story failed editorial gate after rewrite: {second_score}/10")
    second["rewrite_attempted"]=True
    second["initial_score"]=score
    return revised,second
def self_test():
    sample=norm("The shop owner offered Max one free balloon if he carried it home carefully. Max tied it to his wrist, then decided the safest place was inside his jacket. The balloon pushed the jacket open, so he added a belt. The belt trapped his arms. He tried walking sideways, but the balloon kept turning him toward the shop. A gust pulled him backwards, and he grabbed a street sign to stop. The owner ran over and cut the string. Max thanked him for saving the balloon. Then the owner pointed at the tiny receipt still tied to Max's wrist.")
    parts=split_exact(sample,8)
    assert len(parts)==11 and " ".join(parts)==sample and sum(wc(x) for x in parts)==wc(sample)
    print(f"[V2 SELF-TEST] PASS: {len(parts)} exact excerpts, {wc(sample)} words")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); args=ap.parse_args()
    if args.self_test: self_test(); return
    if not KEY: raise SystemExit("GROQ_API_KEY is missing")
    RUN.mkdir(parents=True,exist_ok=True); client=Groq(api_key=KEY)
    user="Write one fresh original Chispa-Pifia story of 180–205 spoken words, ideally 188–198. Immediate visual hook, clear goal, at least three distinct causal escalations, and a satisfying final punchline. Prioritize originality, clarity and a strong ending over random absurdity."
    topic=os.getenv("RELIC_LOOP_TOPIC","").strip()
    if topic: user+=" Optional story seed (use only if it improves the story): "+topic
    history_path=ROOT/"state"/"story_history.json"
    try:
        old=json.loads(history_path.read_text()); titles=[norm(x) for x in old if norm(x)]
        if titles: user+=" Avoid repeating these recent titles: "+"; ".join(titles[-12:])
    except Exception: pass
    story=None; review=None; last=None
    for attempt in range(1,4):
        try:
            raw=call(client,MODEL,STORY_SYSTEM,user,5200,0.82 if attempt==1 else 0.68)
            candidate=validate_story(raw)
            candidate,editorial=review_story(client,candidate)
            story,review=candidate,editorial
            break
        except Exception as e:
            last=e
            print(f"[V2 STORY] candidate {attempt}/3 rejected: {type(e).__name__}: {e}")
            if attempt<3: time.sleep(3)
    if story is None:
        raise SystemExit(f"No story passed editorial quality gate after 3 candidates: {last}")
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
    print(f"[V2 STORY] READY: {package['title']} | {total} words | editorial score={review.get('score')}/10 | {len(scenes)} sentence-aware scenes")
if __name__=="__main__": main()
