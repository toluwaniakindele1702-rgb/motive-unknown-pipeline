#!/usr/bin/env python3
"""Live, non-publishing Creative AI image-provider integration test.

Renders three continuous frames through the same Raphael -> Cloudflare router
used by Relic Loop, with Creative AI prompt construction and image QA enabled.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw

import cloudflare_relic_loop_images_peanut as providers
import cloudflare_relic_loop_images as base
from creative_ai_image_generator import ImageGenerationBrief, render_scene

OUT = Path("state/creative_ai_live_test")
OUT.mkdir(parents=True, exist_ok=True)

def main() -> None:
    providers.PROVIDER_EVENTS.clear()
    for key in providers.PROVIDER_COUNTS:
        providers.PROVIDER_COUNTS[key] = 0

    refs = base._load_reference()
    if len(refs) < 1:
        raise RuntimeError("No canonical Chispa-Pifia visual reference was loaded")

    protagonist = (
        "Milo: the exact approved Chispa-Pifia stickman; large smooth pale round head, "
        "same simple expressive face and eye spacing as canonical reference, compact "
        "dark charcoal hoodie, dark trousers, black-and-white high-top sneakers; "
        "unchanged head/body proportions, silhouette and outline weight"
    )
    setting = (
        "one small warm illustrated 2D home kitchen; pale mint cabinets, cream tiled "
        "backsplash, wooden counter, same window and same lighting in every frame"
    )
    style = (
        "match the supplied canonical Chispa-Pifia character and style references exactly; "
        "polished clean 2D storytime cartoon, hand-drawn bold outlines, expressive poses, "
        "coherent fully illustrated background and props, restrained cel shading"
    )
    frames = [
        ImageGenerationBrief(
            scene_number=1,
            narration_beat="Milo proudly balances an absurdly tall stack of pancakes on a plate on the kitchen counter; the stack is beginning to wobble.",
            protagonist_identity=protagonist, setting=setting, visual_style=style,
            previous_scene_state="Opening shot. Milo stands behind the wooden kitchen counter. A tall pancake stack is on a white plate directly in front of him.",
            shot_direction="vertical medium-wide establishing shot, show Milo, counter and full pancake stack clearly",
            important_props=("white plate", "tall pancake stack", "wooden counter"),
        ),
        ImageGenerationBrief(
            scene_number=2,
            narration_beat="The pancake stack slides sideways, and Milo frantically tries to steady it with a spatula.",
            protagonist_identity=protagonist, setting=setting, visual_style=style,
            previous_scene_state="Same kitchen, same counter, same outfit and lighting. The white plate and pancake stack remain in front of Milo; the stack is now tipping sideways. Keep all established objects consistent.",
            shot_direction="dynamic side three-quarter medium action shot, clear view of Milo's spatula and tipping pancakes",
            important_props=("same white plate", "same pancake stack now tipping", "same spatula", "same wooden counter"),
        ),
        ImageGenerationBrief(
            scene_number=3,
            narration_beat="One pancake lands perfectly on top of Milo's round head like a hat; Milo freezes with a stunned expression.",
            protagonist_identity=protagonist, setting=setting, visual_style=style,
            previous_scene_state="Direct continuation in the identical kitchen. Milo wears the same dark outfit. The pancake stack has toppled; one pancake has landed flat on Milo's head. Keep the same counter, plate, cabinets and window.",
            shot_direction="comedic medium close-up with Milo's full head and pancake clearly visible, kitchen background still recognizable",
            important_props=("single pancake resting flat on Milo's head", "same white plate", "remaining pancakes on counter"),
        ),
    ]

    reports = []
    previous_state = ""
    for brief in frames:
        if previous_state:
            brief = ImageGenerationBrief(
                scene_number=brief.scene_number,
                narration_beat=brief.narration_beat,
                protagonist_identity=brief.protagonist_identity,
                setting=brief.setting,
                visual_style=brief.visual_style,
                previous_scene_state=previous_state + " " + brief.previous_scene_state,
                shot_direction=brief.shot_direction,
                important_props=brief.important_props,
                avoid=brief.avoid,
            )
        path = OUT / f"creative_ai_scene_{brief.scene_number:02d}.png"
        image_bytes, report = render_scene(
            brief=brief,
            provider_router=providers._route_image,
            references=refs,
            output_path=path,
        )
        report["file"] = str(path)
        report["provider_counts_so_far"] = dict(providers.PROVIDER_COUNTS)
        reports.append(report)
        previous_state = brief.previous_scene_state
        print("[CREATIVE AI LIVE TEST] " + json.dumps(report, ensure_ascii=False))

    # Build a contact sheet for human visual review.
    thumbs = []
    for item in reports:
        with Image.open(item["file"]) as source:
            thumb = ImageOps.contain(source.convert("RGB"), (360, 640))
            canvas = Image.new("RGB", (380, 690), "white")
            canvas.paste(thumb, ((380-thumb.width)//2, 12))
            ImageDraw.Draw(canvas).text((12, 660), Path(item["file"]).name, fill="black")
            thumbs.append(canvas)
    sheet = Image.new("RGB", (380 * len(thumbs), 690), "#dddddd")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, (index * 380, 0))
    sheet.save(OUT / "contact_sheet.jpg", quality=92)

    manifest = {
        "test_only": True,
        "publishing": False,
        "provider_priority": ["raphael", "cloudflare"],
        "provider_counts": dict(providers.PROVIDER_COUNTS),
        "provider_events": providers.PROVIDER_EVENTS,
        "references_loaded": [name for name, _, _ in refs],
        "reports": reports,
        "human_visual_approval_required": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[CREATIVE AI LIVE TEST COMPLETE] " + json.dumps({
        "provider_counts": manifest["provider_counts"],
        "frames": len(reports),
        "contact_sheet": str(OUT / "contact_sheet.jpg"),
        "publishing": False,
    }))

if __name__ == "__main__":
    main()
