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

PREMISE_SYSTEM="""You are a senior comedy-room story editor creating original concepts for CHISPA-PIFIA, a fast, visual 2D stickman comedy Shorts channel.

Invent THREE genuinely different story premises. Each must be built around a relatable, concrete goal and a highly specific comic problem that can be understood instantly without explanation. Avoid generic mishaps, random absurdity, stale internet jokes, school embarrassment, aliens, magic and familiar AI-story templates.

For each concept provide:
- a short working title;
- an opening image/action that would make someone stop scrolling in the first second;
- the protagonist's simple goal;
- the unusual obstacle and why it matters;
- a four-step causal chain: each new complication must happen BECAUSE of the protagonist's previous choice;
- the exact final punchline idea, including the earlier detail it pays off;
- why the premise is fresh and visually funny.

COMEDY DESIGN: Prefer comic irony, overconfidence, escalating commitment, mistaken assumptions, status reversals, precise callbacks and visual contradictions. The protagonist must cause or worsen the situation through understandable decisions. Escalation must become more surprising and more specific, not merely louder or more chaotic. The ending must reframe an earlier detail in a way that makes the whole story funnier.

Rank the concepts strongest first. Do not write the full story. Return valid JSON only: {"concepts":[{"title":"","hook":"","goal":"","obstacle":"","beats":["","","",""],"payoff":"","freshness":""}]}. No extra text.""" 

STORY_SYSTEM="""You are the lead comedy writer for CHISPA-PIFIA. Your job is to turn a strong approved concept into a genuinely funny, highly watchable 50–90 second story—not to produce generic AI story filler.

NON-NEGOTIABLE COMEDY ARCHITECTURE:
1. OPEN ON THE PAYOFF-BAIT: The first line must show an immediately intriguing, funny visual crisis already happening. The viewer should instantly wonder how this happened.
2. CLARITY: Within the first two sentences, make the protagonist's concrete goal and the central obstacle unmistakable. No exposition dump.
3. CHARACTER: Give the protagonist a recognizable comic flaw (overconfidence, literal thinking, penny-pinching, people-pleasing, stubbornness, etc.). Their decisions—not random coincidence—drive the plot.
4. ESCALATION: Build at least THREE clearly different, increasingly costly complications. Each must be a direct consequence of the protagonist's previous attempt to solve the problem. Every beat should change the situation. No repeating the same gag with a different prop.
5. PAYOFF: Plant a small, memorable object/detail early. The final sentence must flip or repay that detail with a precise, surprising, logically earned joke. It must be funny even when heard without visuals. A pun, random twist, character simply falling, or last-minute new character is not a punchline.
6. ECONOMY: Every sentence must hook, clarify, escalate, reveal character, or pay off. Delete any sentence that does none of those jobs.

WRITE FROM THE SELECTED CONCEPT:
- Preserve its strongest hook, causal chain and payoff. You may improve any weak beat, but do not abandon a strong premise without reason.
- Make the opening sentence a vivid action, not a title explanation. No "one day", "have you ever", greeting, biography, channel intro or slow setup.
- Use specific everyday objects, clear physical choices, expressive reactions and simple locations that can be illustrated cleanly in 2D.
- Use dialogue sparingly and naturally; no dialogue labels or screenplay formatting.
- Give supporting characters a distinct purpose; do not add them just to fill space.
- Use varied sentence lengths and spoken rhythm. Sound like a sharp human comedy writer, not a template.
- Avoid clichés, generic filler, repeated sentence openings, obvious rhymes, forced puns and moral lessons.
- Safe, non-graphic, non-sexual comedy only. No dangerous stunts, unsafe imitation or physically implausible actions.

Before returning, silently test: Is the first sentence scroll-stopping? Can a listener explain the goal after two sentences? Are there at least three causal escalations? Does each escalation make the predicament more specific? Is the final line the funniest line and does it pay off an earlier planted detail? If any answer is no, rewrite before responding.

LENGTH: 180–205 spoken words; target 190–200. Narration only. Fit a natural 50–90 second voiceover without padding.

OUTPUT: Return exactly one valid JSON object, no markdown, with ONLY:
{"title":"short curiosity-building title","narration":"complete spoken story","protagonist":"short character name or role","supporting_characters":["zero to two names or roles"],"setting":"main location in a few words"}
No scenes, beat lists, outlines, image prompts, metadata or extra keys."""

REVIEW_SYSTEM="""You are a strict commissioning editor deciding whether a finished comedy Short is publishable. Score hook, originality, first-listen clarity, protagonist agency, causal escalation, visual filmability, rhythm and especially the final punchline. The last sentence must be a genuinely funny, specific payoff set up by an earlier detail; a pun alone, weak callback, surprising event, or abrupt ending is NOT a strong punchline. Weak ending means overall score cannot exceed 6. Fewer than three distinct causal escalations means score cannot exceed 6. Dangerous stunts cannot pass. 8 is the minimum publishable score; 7 is borderline and MUST be rejected. Be conservative. Return JSON only: {"score":1,"hook_score":1,"escalation_score":1,"punchline_score":1,"main_problem":"one sentence","rewrite_instruction":"specific actionable fix"}."""
REWRITE_SYSTEM="""You are a senior comedy script doctor, not a copy editor. The draft has FAILED editorial review. Do not merely polish its wording or preserve a weak premise out of politeness.

First diagnose the actual structural failure from the editor notes and scores. If the hook is weak, replace the opening image and put the crisis in motion immediately. If escalation is weak, rebuild the middle as three or four DIFFERENT cause-and-effect turns, each caused by the protagonist's attempted solution and each making the predicament more specific. If the punchline is weak, discard it and design a stronger final reversal that pays off a concrete detail planted early. If the premise itself is generic, predictable or cannot support a strong ending, replace the entire premise.

Write a complete, original, genuinely funny story. The protagonist must have a clear goal, a comic flaw and agency. Every paragraph must advance the same causal chain. Use precise physical comedy and objects that can be illustrated. The final sentence must be the strongest laugh: a specific, surprising, logically earned payoff, not a pun, random event, summary or description of what happens next. Make the story funny when read aloud with no visuals.

Do not add filler to reach length. Natural spoken English, crisp rhythm, varied sentence lengths, clear referents. Safe, non-graphic comedy; no dangerous stunts. Exactly 180–205 narration words. Return JSON with only title,narration,protagonist,supporting_characters,setting. No commentary."""
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
def call(client,model,system,user,tokens=1500,temp=0.65,allow_plain_text=False):
    # GPT-OSS occasionally ignores JSON-only instructions and returns a complete
    # story as ordinary prose. For story-writing calls only, preserve that usable
    # text and normalize it into the same internal schema instead of discarding it.
    # Reviewer and storyboard calls remain strict JSON.
    r=client.chat.completions.create(model=model,messages=[{"role":"system","content":system},{"role":"user","content":user}],
      temperature=temp,max_completion_tokens=tokens)
    choice=r.choices[0]
    content=(choice.message.content or "").strip()
    try:
        return parse_json(content)
    except Exception as exc:
        finish=getattr(choice,"finish_reason",None)
        usage=getattr(r,"usage",None)
        completion=getattr(usage,"completion_tokens",None) if usage else None
        print(f"[V2 MODEL] invalid JSON model={model} finish_reason={finish} completion_tokens={completion} content_preview={content[:500]!r}")
        if allow_plain_text and content:
            print("[V2 MODEL] recovering non-JSON writer response as narration text")
            return {"_plain_text":content}
        raise ValueError(f"Model returned invalid/empty JSON (finish_reason={finish}, completion_tokens={completion}): {exc}") from exc
def plain_text_story(raw):
    text=norm(raw.get("_plain_text"))
    # Remove a short standalone title/header if the model supplied one.
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    if len(lines)>1 and len(lines[0].split())<=10 and not re.search(r"[.!?]$",lines[0]):
        text=norm(" ".join(lines[1:]))
    words=spans(text)
    if len(words)>205:
        # Trim only at a sentence boundary; never cut a sentence in half.
        eligible=[m.end() for m in re.finditer(r"[.!?][\"') ]?(?=\s|$)",text)
                 if wc(text[:m.end()])<=205]
        if eligible:
            text=text[:eligible[-1]].strip()
        else:
            text=" ".join(text.split()[:205])
    title=norm(" ".join(text.split()[:7])).strip(" .,!?:;—-") or "A Stickman Misadventure"
    match=re.search(r"\bStickman\s+([A-Z][a-z]+)\b",text)
    protagonist=match.group(1) if match else "The stickman"
    return {"title":title[:90],"narration":text,"protagonist":protagonist,
            "supporting_characters":[],"setting":"Everyday location"}
def normalize_writer_response(raw):
    if isinstance(raw,dict) and raw.get("_plain_text"):
        raw=plain_text_story(raw)
    return validate_story(raw)
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
    """Grade once, allow one rewrite, then return the best available draft even below threshold."""
    def assess(candidate):
        try:
            result=call(client,REVIEW_MODEL,REVIEW_SYSTEM,
                json.dumps({"title":candidate["title"],"narration":candidate["narration"]}),1800,0.1)
        except Exception as exc:
            result={"score":0,"hook_score":0,"escalation_score":0,"punchline_score":0,
                    "main_problem":f"Reviewer unavailable: {type(exc).__name__}: {exc}",
                    "rewrite_instruction":"Strengthen the opening hook, causal escalation and earned final punchline."}
        def rating(key):
            try: return max(0,min(10,int(result.get(key,0))))
            except (TypeError,ValueError): return 0
        scores=tuple(rating(k) for k in ("score","hook_score","escalation_score","punchline_score"))
        result["score"]=scores[0]
        print(f"[V2 REVIEW] overall={scores[0]}/10 hook={scores[1]}/10 escalation={scores[2]}/10 punchline={scores[3]}/10: {norm(result.get('main_problem'))}")
        return result,all(x>=8 for x in scores),scores
    first_review,passed,first_scores=assess(story)
    first_review["initial_score"]=first_scores[0]
    first_review["rewrite_attempted"]=False
    if passed:
        first_review["grading_passed"]=True
        return story,first_review,first_scores
    instruction=norm(first_review.get("rewrite_instruction")) or "Improve the weakest dimensions with a clear hook, safe causal escalation and earned final payoff."
    print(f"[V2 REVIEW] one rewrite attempt allowed; first scores={first_scores}")
    try:
        rewritten=validate_story(call(client,MODEL,REWRITE_SYSTEM,
            json.dumps({"current_story":story,"editor_main_problem":norm(first_review.get("main_problem")),
                "editor_rewrite_instruction":instruction,
                "failed_scores":{"overall":first_scores[0],"hook":first_scores[1],"escalation":first_scores[2],"punchline":first_scores[3]},
                "required_action":"Fix the weakest dimensions first; preserve safe, filmable cause and effect."}),
            5200,0.68))
        second_review,second_passed,second_scores=assess(rewritten)
        second_review["initial_score"]=first_scores[0]
        second_review["rewrite_attempted"]=True
        second_review["grading_passed"]=second_passed
        if (second_scores[0],sum(second_scores[1:])) > (first_scores[0],sum(first_scores[1:])):
            print(f"[V2 REVIEW] rewrite selected; scores improved from {first_scores} to {second_scores}")
            return rewritten,second_review,second_scores
        print(f"[V2 REVIEW] original retained; rewrite did not improve score ({first_scores} vs {second_scores})")
    except Exception as exc:
        print(f"[V2 REVIEW] rewrite failed; retaining original: {type(exc).__name__}: {exc}")
    first_review["grading_passed"]=False
    first_review["rewrite_attempted"]=True
    return story,first_review,first_scores
def self_test():
    sample=norm("The shop owner offered Max one free balloon if he carried it home carefully. Max tied it to his wrist, then decided the safest place was inside his jacket. The balloon pushed the jacket open, so he added a belt. The belt trapped his arms. He tried walking sideways, but the balloon kept turning him toward the shop. A gust pulled him backwards, and he grabbed a street sign to stop. The owner ran over and cut the string. Max thanked him for saving the balloon. Then the owner pointed at the tiny receipt still tied to Max's wrist.")
    parts=split_exact(sample,8)
    assert len(parts)==8 and " ".join(parts)==sample and sum(wc(x) for x in parts)==wc(sample)
    print(f"[V2 SELF-TEST] PASS: {len(parts)} exact excerpts, {wc(sample)} words")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); args=ap.parse_args()
    if args.self_test: self_test(); return
    if not KEY: raise SystemExit("GROQ_API_KEY is missing")
    RUN.mkdir(parents=True,exist_ok=True); client=Groq(api_key=KEY)
    user="Write one fresh original Chispa-Pifia story of 180–205 spoken words, ideally 188–198. Immediate visual hook, clear goal, at least three distinct causal escalations, and a satisfying final punchline. Prioritize originality, clarity and a genuinely funny earned final punchline over random absurdity. Do not use a weak pun as the entire ending."
    topic=os.getenv("RELIC_LOOP_TOPIC","").strip()
    if topic: user+=" Optional story seed (use only if it improves the story): "+topic
    history_path=ROOT/"state"/"story_history.json"
    try:
        old=json.loads(history_path.read_text()); titles=[norm(x) for x in old if norm(x)]
        if titles: user+=" Avoid repeating these recent titles: "+"; ".join(titles[-12:])
    except Exception: pass
    candidates=[]; last=None
    for attempt in range(1,3):
        try:
            concept_pack=call(client,MODEL,PREMISE_SYSTEM,
                json.dumps({"channel":"Chispa-Pifia","recent_titles":titles[-12:] if 'titles' in locals() else [],
                            "creative_brief":user},ensure_ascii=False),4200,0.9,allow_plain_text=True)
            # If GPT-OSS ignores the premise JSON schema and writes the story
            # directly, salvage that narration instead of failing the run.
            if concept_pack.get("_plain_text"):
                candidate=normalize_writer_response(concept_pack)
                candidate,editorial,scores=review_story(client,candidate)
                editorial["selected_premise"]="Recovered direct writer response"
                editorial["story_attempt"]=attempt
                editorial["component_scores"]={"overall":scores[0],"hook":scores[1],"escalation":scores[2],"punchline":scores[3]}
                candidates.append((candidate,editorial,scores))
                print(f"[V2 STORY] attempt {attempt}: direct prose recovered as a valid story")
                if editorial.get("grading_passed"): break
                continue
            concepts=concept_pack.get("concepts")
            if not isinstance(concepts,list) or not concepts:
                raise ValueError("Premise room returned no usable concepts")
            selected=next((x for x in concepts if isinstance(x,dict)
                           and len(norm(x.get("hook")))>=12 and len(norm(x.get("payoff")))>=12),None)
            if selected is None:
                raise ValueError("Premise room returned no concept with a concrete hook and payoff")
            print(f"[V2 STORY] story attempt {attempt}/2 selected premise: {norm(selected.get('title'))}")
            raw=call(client,MODEL,STORY_SYSTEM,
                json.dumps({"creative_brief":user,"selected_concept":selected,
                            "instruction":"Write the strongest complete narration possible from this premise. Prioritize a scroll-stopping opening and an earned final punchline."},ensure_ascii=False),
                7000,0.78 if attempt==1 else 0.72,allow_plain_text=True)
            candidate=normalize_writer_response(raw)
            candidate,editorial,scores=review_story(client,candidate)
            editorial["selected_premise"]=norm(selected.get("title"))
            editorial["story_attempt"]=attempt
            editorial["component_scores"]={"overall":scores[0],"hook":scores[1],"escalation":scores[2],"punchline":scores[3]}
            candidates.append((candidate,editorial,scores))
            if editorial.get("grading_passed"):
                print(f"[V2 STORY] attempt {attempt} passed required grading; selecting it")
                break
            print(f"[V2 STORY] attempt {attempt} below grading threshold; retaining candidate for comparison")
        except Exception as exc:
            last=exc
            print(f"[V2 STORY] attempt {attempt}/2 generation failed: {type(exc).__name__}: {exc}")
        if attempt<2:
            time.sleep(2)
    if not candidates:
        raise SystemExit(f"Both permitted story attempts failed to produce a structurally valid story. Last error: {last}")
    if len(candidates)==1:
        story,review,scores=candidates[0]
    else:
        story,review,scores=max(candidates,key=lambda item:(item[2][0],sum(item[2][1:])))
        print(f"[V2 STORY] both attempts completed; selected highest-scoring story with scores={scores}")
    review["grading_passed"]=all(x>=8 for x in scores)
    review["selection_policy"]="highest_scoring_of_two_attempts" if len(candidates)==2 and not review["grading_passed"] else "first_passing_candidate_or_best_available"
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
