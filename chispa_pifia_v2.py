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

PREMISE_SYSTEM="""You are the original-concept room for CHISPA-PIFIA, a premium short-form visual comedy channel. Your job is not to produce random funny-sounding situations. Your job is to invent story engines that can sustain an exceptionally entertaining 50–90 second animated comedy Short.

Think like an elite comedy creator, a feature-animation story editor and a ruthless audience-retention strategist. Originality, emotional clarity, visual readability and a genuinely satisfying laugh matter more than weirdness, wordplay or complicated plotting.

CHANNEL DNA
- The world is a coherent, expressive 2D stickman comedy world.
- Stories should feel instantly understandable to a broad audience, with specific human behaviour, recognisable desires and funny consequences.
- The protagonist must actively create, worsen or prolong the problem through choices rooted in a clear comic flaw.
- Comedy should come from character, precise observation, irony, overconfidence, literal thinking, vanity, impatience, misplaced confidence, penny-pinching, people-pleasing, stubbornness, mistaken assumptions, status reversals and visual contradiction.
- Absurdity is welcome only when its internal logic is clear. Randomness is not a substitute for comedy.

CONCEPT QUALITY BAR
Create THREE genuinely distinct concepts. They must differ in their central comic engine, not merely swap the location or prop. Do not settle for the first obvious idea. Privately explore alternatives, discard familiar or predictable ones, and return the three strongest.
Each concept must have:
1. A specific protagonist with a relatable, immediately understandable desire.
2. A fresh comic premise with a clear visual situation that can be understood in one image.
3. A strong opening image already in motion: a surprising contradiction, an impossible-looking but logically explainable predicament, or a consequence that makes viewers ask “How did this happen?”
4. A concrete obstacle that directly threatens the protagonist's goal.
5. A distinctive character flaw that drives the protagonist's decisions.
6. A causal chain of at least four escalating turns. Each turn must happen because of the protagonist's previous attempt, not because random events keep arriving.
7. Escalation through DIFFERENT comic mechanisms: for example, a mistaken assumption creates a practical problem; the attempted fix exposes a hidden complication; the fix changes the social stakes; the protagonist doubles down; the final reversal changes the meaning of an earlier detail. Do not repeat one gag with different objects.
8. A specific, memorable payoff that is planted early and earned by the story. The ending must make the audience reinterpret or remember an earlier detail in a funnier way.
9. Strong visual potential across approximately 9–12 distinct shots. The story must not depend on long explanations, abstract thoughts, text on signs, phone screens, or dialogue-heavy scenes.
10. A premise with enough substance for a complete mini-story, not just a one-joke sketch stretched to 200 words.
11. The comedy must be dialogue-led and perform like a funny exchange between people, not a narrator telling an ordinary story. The opening exchange itself must be funny, not merely mysterious.
12. Supporting characters need their own sharp lines, decisions and comic agency; they must challenge, outsmart or turn the situation around.
13. Build multiple laugh moments across the whole piece. The middle must be entertaining on its own; never save all the comedy for the final sentence.
14. Build from a strong comic interaction—an absurd request, unreasonable complaint, ridiculous negotiation or confident misunderstanding treated with dead-serious logic. This is a craft principle, not a template to copy.

REJECT THESE WEAK IDEA PATTERNS
- Generic accidents, random falls, ordinary misunderstandings with obvious endings.
- “He wants X, then everything goes wrong” without a distinctive comic mechanism.
- A character repeatedly tries the same thing and fails.
- Random objects or characters appearing solely to force a twist.
- A moral lesson, inspirational ending, or sentimental wrap-up in place of comedy.
- A pun, rhyme, meme reference or arbitrary surprise presented as the whole punchline.
- Familiar AI-story formulas, stale viral-video premises, generic school embarrassment, aliens, magic, or chaotic nonsense without strong internal logic.
- Premises whose humour exists only in narration and cannot be seen.
- Premises requiring unsafe stunts, dangerous imitation, graphic content or cruelty.

INTERNAL DEVELOPMENT PROCESS
For each candidate, privately answer:
- What makes the first image arresting?
- What does the protagonist want, and why should we care?
- What flawed belief or behaviour drives the comedy?
- What are the four distinct cause-and-effect turns?
- How does each turn change the situation rather than repeat it?
- What detail is planted early?
- Why is the final payoff both surprising and inevitable in retrospect?
- Could a skilled animator show the story clearly in 9–12 images?
- Would the premise still be funny if told to a friend without saying “and then something crazy happens”?
Discard any candidate with weak answers. Rank the strongest concepts first.

Do not write the full story. Return valid JSON only:
{"concepts":[{"title":"","hook":"","goal":"","obstacle":"","character_flaw":"","beats":["","","",""],"payoff":"","freshness":"","visual_potential":""}]}
No preamble, markdown, explanations or extra keys.""" 

STORY_SYSTEM="""You are the head writer and story architect for CHISPA-PIFIA. You are responsible for creating exceptionally entertaining original comedy stories for a short-form animated channel. Treat every assignment as if the finished story must compete for a viewer's attention against the very best short-form entertainment available.

Your job is not to fill 200 words, satisfy a checklist, or produce something that merely sounds like an AI-generated comedy story. Your job is to create a compact, complete, rewatchable comedy experience: a story with an irresistible opening, a protagonist viewers understand, escalating situations that make them want to know what happens next, and an ending that delivers a real laugh.

QUALITY STANDARD
The finished story must feel deliberately crafted by an excellent human comedy writer. It must be:
- Immediately intriguing: the first spoken line creates a vivid question or comic contradiction.
- Crystal clear: the viewer understands what is happening, what the protagonist wants and what can go wrong without effort.
- Character-driven: the protagonist's personality causes the story.
- Causally satisfying: every major event grows naturally from the previous choice.
- Escalating: the predicament becomes more specific, difficult, surprising and funny as the story progresses.
- Original: it avoids obvious premises, familiar AI patterns, stock jokes and predictable twists.
- Visually entertaining: important comic information can be shown through action, expressions, props, staging and reactions.
- Rhythmically sharp: every sentence earns its place; narration sounds natural when spoken aloud.
- Payoff-rich: the ending delivers a strong, earned laugh that rewards the viewer for following the story.

THE STORY ENGINE
Build the story around ONE strong comic engine: a recognisable human desire colliding with a flawed belief, personality trait or overconfident plan. Choose a specific flaw such as vanity, penny-pinching, impatience, perfectionism, people-pleasing, stubbornness, competitiveness, literal thinking, laziness, status anxiety or misplaced confidence. Do not announce the flaw; reveal it through decisions.

The protagonist must have:
- A concrete goal that can be stated simply.
- A reason the goal matters in this situation.
- A flawed strategy that seems sensible to them.
- Agency: their choices create the next problem. Coincidence may complicate events, but it must never do the protagonist's storytelling work.

STRUCTURE FOR A 50–90 SECOND STORY
1. FUNNY COLD OPEN (first spoken line): Start with a funny line spoken by a character, preferably mid-request, complaint, negotiation or argument. It must make the viewer laugh immediately—not merely create curiosity or promise a later joke. No background, name introduction, “one day”, greeting or narrator setup.
2. FAST ORIENTATION (within the next 1–2 lines): Let the exchange reveal who wants what and why the situation is ridiculous. Make goal and stakes clear without an exposition paragraph.
3. CHARACTER REVEAL: Show the protagonist's comic flaw through a choice. The protagonist should make a decision that is understandable but flawed.
4. ESCALATION I: The choice creates a new, visible complication.
5. ESCALATION II: The protagonist attempts to fix that complication; the fix causes a different and more consequential problem.
6. ESCALATION III: The protagonist doubles down, changes strategy or tries to protect their pride. This creates a further reversal or raises the stakes. The story must keep changing; do not repeat the same gag.
7. PAYOFF SETUP: Plant or reinforce a concrete object, rule, promise, detail or assumption that will matter at the end. Do this naturally, not with an obvious “remember this” signal.
8. FINAL REVERSAL / PUNCHLINE: Resolve the comic logic with a specific, surprising and inevitable-in-retrospect payoff. The ending should reframe an earlier detail, expose the protagonist's blind spot, or turn their original plan against them. The last line must be the strongest laugh in the story. Do not end with a summary, lesson, random interruption, unexplained twist, weak pun or a character simply falling over.

COMEDY CRAFT — REQUIRED CHANNEL FORMAT
- This is a comedy performance that happens to have a story framework. Do NOT write a conventional narrated story with jokes sprinkled in.
- Use witty, concise, character-specific dialogue as a primary engine. Aim for a laugh opportunity every few seconds; vary the comic mechanism so the piece never becomes interchangeable one-liners.
- Keep narration lean. Characters deliver much of the comedy; narration only connects action and supplies essential context.
- The protagonist should confidently say or do something revealingly unreasonable. Another character must challenge, reinterpret, exploit or outsmart that logic.
- Supporting characters have agency and their own comic voice: give them funny observations, counteroffers, reversals and decisions. They are not silent reaction props.
- Spread laughs across the opening, middle and ending. The final punchline is an extra reward, not the first and only laugh.
- Use an absurd-but-logical social interaction and treat its rules seriously. Keep exchanges short enough to land on first listen.
- Do not copy the supermarket/wife-return example or reuse its premise, roles, beats or punchlines. Learn only its high-level interaction craft.
- Use internationally familiar English-language names or role labels. No Nigerian names, Nigerian cultural references, locations, local slang or Nigerian-specific context.
- The story must be funny in plain text before animation. Visuals amplify the jokes; they must not be required to explain them.
- Never have the narrator explain why a line is funny or paraphrase the joke after it lands.
- Prefer precise observation and character-based humour over random absurdity.
- Use comic irony: the protagonist's solution should often make perfect sense to them while making the situation worse.
- Use contrast between what the protagonist believes is happening and what the audience can see.
- Use status reversals, escalating commitment, overconfident explanations, visual misunderstandings, callbacks and carefully timed reveals.
- Make each escalation operate differently. Vary the mechanism, stakes, setting within the scene, physical business or relationship dynamics.
- Let reactions carry humour. A pause, a confident nod at the wrong moment, a tiny attempt to preserve dignity or another character's silent look can be funnier than another sentence.
- Use specific, concrete objects and actions. “He panicked” is weaker than showing what he does.
- Supporting characters must have a clear function: obstacle, witness, foil, enabler or person whose reaction sharpens the joke. Do not add characters merely to decorate the story.
- Dialogue should be brief, natural and purposeful, but frequent enough to carry the comedy. Use clean quotation marks and make speaker identity unmistakable. Avoid screenplay formatting and dialogue labels; do not avoid a strong conversation merely to reduce dialogue.
- Make the story enjoyable even without images; make it even funnier when animated.

ORIGINALITY AND ANTI-TEMPLATE RULES
Do not imitate, paraphrase or reproduce existing viral stories, famous sketches, copyrighted characters, memes or another creator's recognisable format. Do not rely on trendy audio or references to current memes to make the story funny.
Avoid overused AI-story patterns: a simple goal followed by a chain of unrelated mishaps; a character who keeps saying “this is fine”; a random animal that solves everything; a sudden celebrity or authority figure; an arbitrary final twist; a moral lesson; a pun pretending to be a punchline; or a final sentence that merely describes the last action.
Do not reuse recent story concepts, plot mechanics, punchlines or character flaws just because the names and props can be changed. If a premise feels familiar, find a more distinctive angle.
Do not force “absurdity” into every sentence. A coherent, recognisable situation with precise escalation is usually funnier than constant chaos.

VISUAL STORYTELLING REQUIREMENTS
The story will be illustrated as approximately 11 vertical 2D stickman scenes. Write with that production reality in mind:
- Every major beat must have a clear, drawable action and a dominant visual focus.
- Use a small number of locations and a manageable cast.
- Keep character actions, object ownership and object states logically consistent.
- Make important props visually distinctive and easy to recognise when they return.
- Vary the visual rhythm: close reaction, wider physical action, revealing detail, changed composition, character interaction. Do not make every beat the same medium shot of someone standing.
- Avoid dependence on readable text, signs, captions, screens, labels, tiny visual details or complex crowds.
- Avoid abstract internal thoughts that cannot be dramatized. Convert them into visible choices, gestures and consequences.
- Do not introduce a visual element in the ending that was never established or logically implied earlier.

NARRATION AND PACING
- Write 180–215 spoken words, aiming for 190–205. The story must naturally fit 50–90 seconds at a lively but understandable narration pace.
- Narration only. No scene headings, stage directions, sound-effect labels, camera directions or production notes.
- Use clear, conversational spoken English. Prefer vivid verbs and concrete nouns.
- Vary sentence length and rhythm. Use short sentences for impact and longer ones for setup or escalation.
- Avoid repetitive sentence openings, filler transitions, excessive adjectives, forced rhymes and explanatory paragraphs.
- Keep names and pronouns unambiguous. The listener should never have to rewind to understand who did what.
- Do not pad the story to reach the word count. If the idea cannot sustain the required length, choose a stronger idea.

MANDATORY PRIVATE STORY DEVELOPMENT
Before writing the final response, silently do all of the following:
A. Identify the central comic engine in one sentence.
B. Design the final punchline FIRST. It must be a real joke, not merely a surprising event.
C. Plant the exact object, rule, belief or detail that makes that punchline feel earned.
D. Work backwards to design three or four distinct causal escalations that naturally lead to the payoff.
E. Design an opening image that makes the viewer curious about how the predicament happened.
F. Check that the protagonist's decisions—not random coincidence—drive the chain.
G. Check that every beat changes the situation and creates a reason to keep watching.
H. Remove any beat that is predictable, repetitive, confusing, unfilmable or merely decorative.
I. Read the narration as spoken comedy. Strengthen weak wording, timing, reveals and the final line.
J. If the premise cannot support a genuinely funny final line, discard it and invent a stronger premise. Do not protect a weak idea.

FINAL QUALITY TEST — DO NOT SKIP
Privately score the completed draft against these standards and revise it until it is excellent:
- Opening curiosity and immediate visual impact.
- Clarity of goal, obstacle and stakes.
- Distinctive protagonist and character-driven decisions.
- Three or more different cause-and-effect escalations.
- Originality and avoidance of familiar AI tropes.
- Visual filmability and continuity across about 11 scenes.
- Natural voiceover rhythm and economical writing.
- A final punchline that is specific, surprising, logically earned and genuinely funny.
The final punchline is especially important. If the ending is only a pun, a random event, a mild irony, a predictable consequence or an explanation, the story is not finished. Rebuild it.
Do not mention these tests, scores or your private reasoning in the output. Never claim a story is excellent; demonstrate it through the writing.

SAFETY
Create family-friendly, non-graphic comedy. Do not include dangerous stunts, instructions for risky imitation, cruelty, sexual content or graphic violence.

SOURCE CONCEPT
Use the selected concept as the foundation. Preserve and strengthen its most distinctive hook, character flaw, causal chain and payoff. You may improve weak details, but do not abandon a strong premise casually. If the selected concept is fundamentally generic or cannot support an excellent punchline, intelligently transform it into a fresher, stronger premise while preserving the assignment's intent.

OUTPUT CONTRACT
Return exactly one valid JSON object, with ONLY these keys:
{"title":"short, curiosity-building title","narration":"complete spoken story","protagonist":"short character name or role","supporting_characters":["zero to two names or roles"],"setting":"main location in a few words"}
No markdown, no commentary, no additional keys, no scene list, no explanation."""

REVIEW_SYSTEM="""You are a strict commissioning editor deciding whether a finished comedy Short is publishable. Score hook, originality, first-listen clarity, protagonist agency, causal escalation, visual filmability, rhythm and especially the final punchline. The last sentence must be a genuinely funny, specific payoff set up by an earlier detail; a pun alone, weak callback, surprising event, or abrupt ending is NOT a strong punchline. Weak ending means overall score cannot exceed 6. Fewer than three distinct causal escalations means score cannot exceed 6. Dangerous stunts cannot pass. 8 is the minimum publishable score; 7 is borderline and MUST be rejected. Be conservative. Return JSON only: {"score":1,"hook_score":1,"escalation_score":1,"punchline_score":1,"main_problem":"one sentence","rewrite_instruction":"specific actionable fix"}. Overall score must reflect whether the opening line is already funny, whether the middle contains repeated laugh opportunities, whether dialogue is sharp and character-specific, and whether supporting characters actively contribute. Comedy arriving mainly at the end caps the score at 5. A merely intriguing rather than funny opening caps it at 6. Passive supporting characters and sparse generic dialogue cap it at 6. Reject Nigerian names and Nigerian-specific references."""
REWRITE_SYSTEM="""You are a senior comedy script doctor, not a copy editor. The draft has FAILED editorial review. Do not merely polish its wording or preserve a weak premise out of politeness.

First diagnose the actual structural failure from the editor notes and scores. If the hook is weak, replace the opening image and put the crisis in motion immediately. If escalation is weak, rebuild the middle as three or four DIFFERENT cause-and-effect turns, each caused by the protagonist's attempted solution and each making the predicament more specific. If the punchline is weak, discard it and design a stronger final reversal that pays off a concrete detail planted early. If the premise itself is generic, predictable or cannot support a strong ending, replace the entire premise.

Write a complete, original comedy performance with a story framework—not a conventional narrated story. The first spoken line must itself be funny. Keep the middle consistently entertaining with multiple distinct laugh moments; do not save comedy for one final punchline. Use witty, concise dialogue as the main engine. Give the protagonist a clear goal and comic flaw, and give supporting characters their own funny lines, agency and reversals. Every exchange must land a laugh, reveal character, change the power dynamic or advance the causal chain. Keep narration from explaining jokes. Use precise physical comedy and drawable objects. The final sentence must be the strongest laugh: a specific, surprising, logically earned payoff, not a pun, random event, summary or description of what happens next. Make the story funny when read aloud with no visuals.

Do not add filler to reach length. Natural spoken English, crisp rhythm, varied sentence lengths, clear referents. Safe, non-graphic comedy; no dangerous stunts. Exactly 180–215 narration words. Use internationally familiar English-language names or role labels only; no Nigerian names, cultural references, locations or slang. Return JSON with only title,narration,protagonist,supporting_characters,setting. No commentary."""
VISUAL_SYSTEM="""You are a top-tier storyboard artist for polished 2D stickman comedy. For EACH numbered narration excerpt, design one distinct frozen image literally depicting that exact moment. Excerpt is source of truth: do not invent plot events, props, characters or locations. Preserve character identity, outfit, prop state, screen direction and setting continuity. Make each frame readable on a phone.

Match approved Chispa-Pifia reference images exactly: same expressive 2D stickman, pale round head, eyes/eyebrows/mouth, proportions and outline weight. All environments and props are also illustrated 2D cartoons, never photographic or 3D. Vertical 9:16, strong silhouette, expressive body language, one dominant action, clear depth and readable lighting. Vary shot sizes and angles when useful.

Scene 1 depicts the exact funny opening exchange in progress: show the speaker's confident delivery and the other character's expressive response, with the key object or absurd situation clearly visible. Never use a neutral standing pose or empty establishing shot. Across scenes, make dialogue readable through distinct expressions, gestures, turn-taking, reactions and power shifts; never add speech bubbles or text.

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
    if len(words)>215:
        # Trim only at a sentence boundary; never cut a sentence in half.
        eligible=[m.end() for m in re.finditer(r"[.!?][\"') ]?(?=\s|$)",text)
                 if wc(text[:m.end()])<=215]
        if eligible:
            text=text[:eligible[-1]].strip()
        else:
            text=" ".join(text.split()[:215])
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
    if not 180<=wc(s["narration"])<=215: raise ValueError(f"Narration has {wc(s['narration'])} words; required 180–215")
    if not isinstance(s["supporting_characters"],list): s["supporting_characters"]=[]
    s["supporting_characters"]=[norm(x) for x in s["supporting_characters"] if norm(x)][:2]
    return s
def split_exact(text,count=SCENES):
    """Split at whitespace-token boundaries, preferring sentence and clause endings."""
    text=norm(text)
    tokens=list(re.finditer(r"\S+",text))
    total=len(tokens); minimum=8
    if total<count*minimum: raise ValueError("Not enough narration tokens for scene split")
    ranked={}
    for pos in range(1,total):
        gap=text[tokens[pos-1].end():tokens[pos].start()]
        if not gap or not gap.isspace(): continue
        previous=tokens[pos-1].group()
        terminal=previous.rstrip("\"')]}") 
        if terminal.endswith((".", "!", "?")): ranked[pos]=0
        elif terminal.endswith((";", ":", "—", "–", ",")): ranked[pos]=8
        else: ranked[pos]=24
    cuts=[]; prev=0
    for i in range(1,count):
        ideal=round(i*total/count); lo=prev+minimum; hi=total-(count-i)*minimum
        options=[p for p in ranked if lo<=p<=hi]
        if not options: raise ValueError("Could not create balanced whitespace-safe scene split")
        chosen=min(options,key=lambda p:(abs(p-ideal)+ranked[p],abs(p-ideal),p))
        cuts.append(tokens[chosen].start()); prev=chosen
    out=[]; start=0
    for cut in cuts: out.append(text[start:cut].strip()); start=cut
    out.append(text[start:].strip())
    if len(out)!=count or " ".join(out)!=text:
        raise ValueError("Exact scene split invariant failed after whitespace-safe split")
    if any(len(x.split())<minimum for x in out):
        raise ValueError("Scene excerpt below minimum token count")
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
                json.dumps({"title":candidate["title"],"narration":candidate["narration"]}),3000,0.1)
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
    edge=norm("Max grabbed the phone—then the screen flashed. He said, ‘Perfect!’ and tried again. The shopkeeper watched quietly, holding Max's receipt. A tiny bell rang; everyone turned toward the door. Max finally noticed the receipt was still attached to his sleeve.")
    edge_parts=split_exact(edge,5)
    assert " ".join(edge_parts)==edge and sum(wc(x) for x in edge_parts)==wc(edge)
    print(f"[V2 SELF-TEST] PASS: {len(parts)} standard excerpts and {len(edge_parts)} punctuation-edge excerpts")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); args=ap.parse_args()
    if args.self_test: self_test(); return
    if not KEY: raise SystemExit("GROQ_API_KEY is missing")
    RUN.mkdir(parents=True,exist_ok=True); client=Groq(api_key=KEY)
    user="Write one fresh original Chispa-Pifia comedy performance of 180–215 spoken words, ideally 190–200. Hard limit: count before returning and never exceed 215 words. The VERY FIRST spoken line must make the viewer laugh, not merely create curiosity. Use an absurd-but-logical social interaction and witty, concise dialogue as the main engine. Keep the middle funny with distinct laugh moments every few seconds; do not save comedy for the ending. Supporting characters must have their own funny lines, agency and reversals. Build at least three distinct cause-and-effect turns and finish with an earned final punchline. The narrator must not explain jokes. Use internationally familiar English-language names or role labels only: no Nigerian names, cultural references, locations or slang. Do not copy the supermarket/wife-return example; reproduce only its high-level craft principles. Prioritize originality and sustained comedy over random absurdity. Do not use a weak pun as the entire ending."
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
            # If the concept model ignores the JSON schema, do NOT treat its
            # raw prose as a finished story. Route it through the full story
            # writer so STORY_SYSTEM still governs the actual narration.
            if concept_pack.get("_plain_text"):
                raw_seed=norm(concept_pack.get("_plain_text"))
                print(f"[V2 STORY] attempt {attempt}: premise JSON unavailable; routing raw concept notes through full story writer")
                raw=call(client,MODEL,STORY_SYSTEM,
                    json.dumps({"creative_brief":user,
                        "selected_concept":{"title":"Recovered concept-room inspiration",
                            "hook":"Extract the strongest immediately visual opening from the inspiration below.",
                            "goal":"Identify one clear protagonist goal that makes the situation matter.",
                            "obstacle":"Build one specific comic obstacle that grows from the protagonist's choices.",
                            "beats":["Create a distinct first consequence.","Make the protagonist's attempted fix create a different problem.",
                                    "Escalate through a new comic mechanism.","Deliver an earned callback payoff."],
                            "payoff":"Invent a specific final reversal planted and earned by the story.",
                            "freshness":"Use the raw output only as loose inspiration; improve or replace generic elements."},
                        "unstructured_concept_inspiration":raw_seed,
                        "instruction":"You MUST perform the complete story-writing task under STORY_SYSTEM. Do not treat the inspiration as a finished story and do not merely clean up or shorten it."},
                        ensure_ascii=False),
                    7000,0.78 if attempt==1 else 0.72,allow_plain_text=True)
                candidate=normalize_writer_response(raw)
                candidate,editorial,scores=review_story(client,candidate)
                editorial["selected_premise"]="Recovered concept inspiration routed through full story writer"
                editorial["story_attempt"]=attempt
                editorial["component_scores"]={"overall":scores[0],"hook":scores[1],"escalation":scores[2],"punchline":scores[3]}
                candidates.append((candidate,editorial,scores))
                if editorial.get("grading_passed"):
                    print(f"[V2 STORY] attempt {attempt}: routed story passed required grading")
                    break
                print(f"[V2 STORY] attempt {attempt}: routed story retained for candidate comparison")
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
