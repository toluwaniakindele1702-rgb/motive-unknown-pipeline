#!/usr/bin/env python3
"""Build a contact sheet of generated frames for visual QA."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
ROOT=Path(__file__).resolve().parent
RUN=ROOT/"state"/"current_run"
def main():
    story=json.loads((RUN/"relic_loop_short.json").read_text(encoding="utf-8"))
    scenes=story.get("scenes") or []
    if not scenes: raise SystemExit("No scenes for contact sheet")
    cols=3; cw,ch,lh,margin=300,535,34,18
    rows=(len(scenes)+cols-1)//cols
    sheet=Image.new("RGB",(margin+cols*(cw+margin),margin+rows*(ch+lh+margin)),"#eeeeee")
    draw=ImageDraw.Draw(sheet)
    try: font=ImageFont.truetype("DejaVuSans.ttf",18)
    except OSError: font=ImageFont.load_default()
    for i,scene in enumerate(scenes):
        p=RUN/"images"/f"beat_{int(scene['number']):03d}.png"
        if not p.exists(): raise SystemExit(f"Missing image {p}")
        with Image.open(p) as im:
            thumb=im.convert("RGB"); thumb.thumbnail((cw,ch),Image.Resampling.LANCZOS)
        x=margin+(i%cols)*(cw+margin); y=margin+(i//cols)*(ch+lh+margin)
        sheet.paste(thumb,(x+(cw-thumb.width)//2,y))
        draw.text((x,y+ch+5),f"SCENE {scene['number']:02d}",fill="#111111",font=font)
    out=RUN/"contact_sheet.jpg"; sheet.save(out,quality=90,optimize=True)
    print(f"[CONTACT SHEET] {out} ({len(scenes)} frames)")
if __name__=="__main__": main()
