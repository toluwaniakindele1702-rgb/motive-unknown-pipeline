from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
REF_DIR = ROOT / "assets" / "visual_references"
REF_FILES = [
    REF_DIR / "file_00000000078881f486600a6d2d8ccc89.png",
    REF_DIR / "file_000000008bfc81f480840c507c90542b.png",
]
MODEL = "@cf/black-forest-labs/flux-2-dev"
ACCOUNT = os.environ["CLOUDFLARE_ACCOUNT_ID"]
TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
OUT = ROOT / "flux2_dev_test"
OUT.mkdir(exist_ok=True)

PROMPTS = [
    "Use reference image 0 only to preserve the recognizable round-headed stylized stickman identity; use reference image 1 only for the general illustrated environment quality. Create ONE single full-frame vertical 9:16 2D cartoon animation shot: the same stickman, wearing a vivid red hoodie, dark navy-blue trousers and green-and-white sneakers, panicking while trying to close an overfilled blue suitcase in a messy bedroom. Large smooth white circular head, simple expressive black eyes and eyebrows, clean thick black outlines, expressive pose, polished cel shading, colorful detailed cartoon background. Entire character must remain a stylized stickman, not a realistic human. Exactly two arms and two legs, anatomically clear pose, five fingers at most per visible hand. No text, no letters, no numbers, no speech bubbles, no panels, no collage, no model sheet, no logos, no watermark.",
    "Use reference image 0 only to preserve the recognizable round-headed stylized stickman identity; use reference image 1 only for the general illustrated environment quality. Create ONE single full-frame vertical 9:16 2D cartoon animation shot: the same stickman in a red hoodie, navy-blue trousers and green-and-white sneakers, sprinting through an airport terminal while pulling a blue suitcase, looking over his shoulder in panic. Large smooth white circular head, simple expressive black eyes and eyebrows, clean thick black outlines, expressive action pose, polished cel shading, colorful detailed cartoon airport background. Entire character must remain a stylized stickman, not a realistic human. Exactly two arms and two legs, anatomically clear pose, five fingers at most per visible hand. No text, no letters, no numbers, no speech bubbles, no panels, no collage, no model sheet, no logos, no watermark.",
    "Use reference image 0 only to preserve the recognizable round-headed stylized stickman identity; use reference image 1 only for the general illustrated environment quality. Create ONE single full-frame vertical 9:16 2D cartoon animation shot: the same stickman in a red hoodie, navy-blue trousers and green-and-white sneakers, standing frozen in shock in a small apartment doorway after realizing he brought the wrong suitcase. Large smooth white circular head, simple expressive black eyes and eyebrows, clean thick black outlines, readable silhouette, polished cel shading, colorful detailed cartoon apartment background. Entire character must remain a stylized stickman, not a realistic human. Exactly two arms and two legs, anatomically clear pose, five fingers at most per visible hand. No text, no letters, no numbers, no speech bubbles, no panels, no collage, no model sheet, no logos, no watermark.",
]

def prepare_reference(path: Path) -> bytes:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((511, 511), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, "JPEG", quality=88, optimize=True)
        return buf.getvalue()

def main() -> None:
    refs = [prepare_reference(p) for p in REF_FILES]
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/{MODEL}"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    for i, prompt in enumerate(PROMPTS, 1):
        files = {
            "prompt": (None, prompt),
            "width": (None, "576"),
            "height": (None, "1024"),
            "steps": (None, "25"),
            "input_image_0": (f"reference_{i}_character.jpg", refs[0], "image/jpeg"),
            "input_image_1": (f"reference_{i}_environment.jpg", refs[1], "image/jpeg"),
        }
        print(f"[TEST] Generating scene {i}/{len(PROMPTS)} with {MODEL}; reference images enabled")
        response = requests.post(url, headers=headers, files=files, timeout=240)
        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[:1000]
            raise RuntimeError(f"Cloudflare rejected scene {i}: HTTP {response.status_code}: {detail}")
        payload = response.json()
        image = (payload.get("result") or {}).get("image")
        if not image:
            raise RuntimeError(f"Cloudflare response for scene {i} did not contain result.image")
        raw = base64.b64decode(image)
        target = OUT / f"scene_{i}.png"
        target.write_bytes(raw)
        with Image.open(target) as im:
            im.verify()
        print(f"[TEST] Saved valid image: {target} ({len(raw)} bytes)")
    print(f"[TEST] Complete. Review all three images in artifact folder: {OUT}")

if __name__ == "__main__":
    main()
