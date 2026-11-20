#!/usr/bin/env python3
"""Pre-production Story Quality AI gate: review, coach, rewrite; never block production."""
from __future__ import annotations
import json, re
from typing import Any

STORY_REVIEW_PROMPT = """You are Story Quality AI, an editor for high-retention 50–90 second animated comedy Shorts.
Judge ONLY the supplied script. Be strict and specific, not encouraging by default.
Score 1–10: opening_hook (first sentence is an immediate, concrete reason to keep watching),
open_loop (a clear story question is planted early), clarity,
escalation (each beat makes the situation more consequential/funnier),
mid_story_rehook (a meaningful new complication or reveal renews curiosity near the middle),
originality, character_voice, voiceover_naturalness (sounds conversational when read aloud),
pacing (no filler), payoff (earned and surprising, planted detail pays off), rewatchability.
A strong script starts in the middle of an unusual situation; uses concrete details and dialogue;
shows rather than explains; has a clear character want; escalates through 3+ connected turns;
includes a genuine mid-story rehook and ends with a concise payoff/callback. Avoid vague mysteries, fake stakes, random twists,
moralizing, generic AI phrases, over-explaining and predictable "it was all a dream" endings.
Return JSON only: {"scores":{...},"overall_score":number,"issues":["..."],"rewrite_brief":"...","verdict":"revise|ready"}.
If any key dimension (opening hook, open loop, escalation, mid-story rehook, voiceover naturalness, payoff, pacing or originality) is below 7, verdict revise.
The rewrite brief should list concrete fixes, preserve the premise if promising, and never add filler.
Do not decide production/upload. Review is advisory; production must proceed after bounded rewrite attempts."""
DIMENSIONS=("opening_hook","open_loop","clarity","escalation","mid_story_rehook","originality","character_voice","voiceover_naturalness","pacing","payoff","rewatchability")

def _parse(raw: str)->dict:
    raw=(raw or "").strip()
    raw=re.sub(r"^\x60{3}(?:json)?\s*|\s*\x60{3}$","",raw,flags=re.I)
    a,b=raw.find("{"),raw.rfind("}")
    if a<0 or b<a: raise ValueError("Story Quality AI returned no JSON")
    value=json.loads(raw[a:b+1])
    if not isinstance(value,dict): raise ValueError("Review response must be an object")
    return value

def review_story(client:Any, model:str, story:dict)->dict:
    response=client.chat.completions.create(model=model,messages=[
        {"role":"system","content":STORY_REVIEW_PROMPT},
        {"role":"user","content":"Review this narration script and title.\n"+json.dumps({"title":story.get("title"),"narration":story.get("narration")},ensure_ascii=False)}
    ],temperature=0.15,max_completion_tokens=1500)
    data=_parse(response.choices[0].message.content or "")
    scores=data.get("scores") if isinstance(data.get("scores"),dict) else {}
    normalized={}
    for dim in DIMENSIONS:
        try: n=float(scores.get(dim))
        except (TypeError,ValueError): n=None
        normalized[dim]=round(n,1) if n is not None and 1<=n<=10 else None
    valid=[v for v in normalized.values() if v is not None]
    overall=data.get("overall_score")
    try: overall=float(overall)
    except (TypeError,ValueError): overall=sum(valid)/len(valid) if valid else None
    issues=data.get("issues",[])
    if isinstance(issues,str): issues=[issues]
    issues=[str(x).strip() for x in issues if str(x).strip()][:8] if isinstance(issues,list) else []
    brief=str(data.get("rewrite_brief","")).strip()
    critical=[normalized[k] for k in ("opening_hook","open_loop","escalation","mid_story_rehook","originality","voiceover_naturalness","pacing","payoff")]
    verdict="revise" if any(x is None or x<7 for x in critical) or data.get("verdict")=="revise" else "ready"
    return {"scores":normalized,"overall_score":round(overall,1) if overall is not None else None,"issues":issues,"rewrite_brief":brief,"verdict":verdict}

def coach_feedback(review:dict)->dict:
    scores=review.get("scores",{})
    low=[f"{k.replace('_',' ')} scored {v}/10; improve this deliberately." for k,v in scores.items() if v is not None and v<8]
    return {"overall_score":review.get("overall_score"),"strengths":[],"improvements":review.get("issues",[])+low,
      "continuity_notes":[],"pacing_notes":[review.get("rewrite_brief","")],"story_notes":["Story Quality AI notes: "+review.get("rewrite_brief","")]}

def run_story_gate(client:Any, model:str, generate, feedback, max_rewrites:int=2,
                   candidate_count:int=2)->tuple[dict,list[dict],dict]:
    """Generate and score multiple candidates, select the strongest, then coach/rewrite."""
    history=[]
    candidates=[]
    count=max(1,min(3,int(candidate_count)))
    for candidate_index in range(count):
        candidate=generate(feedback)
        try:
            review=review_story(client,model,candidate)
            history.append({"candidate":candidate_index+1,**review})
            score=review.get("overall_score")
            if score is None:
                values=[v for v in review.get("scores",{}).values() if v is not None]
                score=sum(values)/len(values) if values else 0.0
            candidates.append((float(score),candidate,review))
        except Exception as exc:
            history.append({"candidate":candidate_index+1,
                            "review_unavailable":type(exc).__name__+": "+str(exc)[:400],
                            "advisory_fallback":True})
            candidates.append((float("-inf"),candidate,None))

    # Preserve generation order if every review failed; otherwise choose the
    # candidate with the highest measured quality score.
    scored=[item for item in candidates if item[2] is not None]
    if scored:
        _,story,review=max(scored,key=lambda item:item[0])
    else:
        _,story,review=candidates[0]
        return story,history,feedback

    for attempt in range(max(0,int(max_rewrites))):
        if review["verdict"]=="ready":
            break
        feedback=coach_feedback(review)
        story=generate(feedback)
        try:
            review=review_story(client,model,story)
            history.append({"rewrite_attempt":attempt+1,**review})
        except Exception as exc:
            history.append({"rewrite_attempt":attempt+1,
                            "review_unavailable":type(exc).__name__+": "+str(exc)[:400],
                            "advisory_fallback":True})
            break
    return story,history,feedback

def self_test():
    sample={"scores":{"opening_hook":5,"clarity":8,"escalation":6,"originality":6,"character_voice":8,"pacing":5,"payoff":6,"rewatchability":6},"issues":["Weak hook"],"rewrite_brief":"Start with the problem.","verdict":"revise"}
    parsed=_parse(json.dumps(sample))
    assert parsed["verdict"]=="revise"
    assert coach_feedback({"scores":sample["scores"],"issues":["Weak hook"],"rewrite_brief":"Fix hook"})["pacing_notes"]
    print("Story Quality AI tests passed")
if __name__=="__main__": self_test()
