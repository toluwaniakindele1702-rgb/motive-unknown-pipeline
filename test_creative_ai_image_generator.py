import unittest
from io import BytesIO
from PIL import Image, ImageDraw

from creative_ai_image_generator import (
    ImageGenerationBrief, build_image_prompt, inspect_image,
    normalize_portrait, render_scene,
)


def fixture_image():
    image = Image.new("RGB", (512, 768), "white")
    draw = ImageDraw.Draw(image)
    # Fine illustrated background detail keeps the fixture representative of a
    # real rendered frame without introducing random/non-deterministic pixels.
    for x in range(0, 512, 8):
        draw.line((x, 0, x, 767), fill=(220 - (x % 80), 230, 240), width=2)
    draw.rectangle((80, 100, 430, 680), outline="black", width=12)
    draw.ellipse((160, 150, 350, 340), fill="#e8d2bb", outline="black", width=8)
    draw.arc((190, 190, 320, 300), 0, 180, fill="black", width=5)
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


class CreativeAIImageTests(unittest.TestCase):
    def setUp(self):
        self.brief = ImageGenerationBrief(
            scene_number=2,
            narration_beat="Milo carefully balances the red lunchbox on one finger.",
            protagonist_identity="Milo, pale round-headed stickman in the same dark hoodie",
            setting="same illustrated kitchen",
            previous_scene_state="red lunchbox was on the counter; kitchen window remains on the left",
            important_props=("red lunchbox",),
        )

    def test_prompt_locks_identity_and_exact_beat(self):
        prompt = build_image_prompt(self.brief)
        self.assertIn("CANONICAL CHARACTER LOCK", prompt)
        self.assertIn("WARDROBE LOCK (MANDATORY)", prompt)
        self.assertIn(self.brief.narration_beat, prompt)
        self.assertIn("CONTINUITY STATE", prompt)
        self.assertIn("ZERO TEXT", prompt)

    def test_rejects_corrupt_and_tiny_images(self):
        self.assertFalse(inspect_image(b"not an image").valid)
        self.assertFalse(inspect_image(b"x" * 100).valid)

    def test_accepts_and_normalizes_portrait_image(self):
        result = normalize_portrait(fixture_image())
        with Image.open(BytesIO(result)) as image:
            self.assertEqual(image.size, (576, 1024))
        self.assertTrue(inspect_image(result).valid)

    def test_renderer_retries_once_with_quality_refinement(self):
        good = fixture_image()
        calls = []
        def router(prompt, refs):
            calls.append(prompt)
            return b"broken" if len(calls) == 1 else good
        rendered, report = render_scene(self.brief, router)
        self.assertTrue(inspect_image(rendered).valid)
        self.assertEqual(len(calls), 2)
        self.assertIn("QUALITY REFINEMENT", calls[1])
        self.assertTrue(report["human_visual_approval_required"])

    def test_successful_frame_is_written_atomically(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "scene.png"
            rendered, _ = render_scene(self.brief, lambda prompt, refs: fixture_image(), output_path=target)
            self.assertTrue(target.exists())
            self.assertEqual(target.read_bytes(), rendered)
            self.assertFalse(target.with_suffix(".png.tmp").exists())


if __name__ == "__main__":
    unittest.main()
