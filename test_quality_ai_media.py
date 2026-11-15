import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quality_ai_media import (
    MAX_IMAGES, _parse_json, build_gemini_client, extract_video_frames,
    inspect_media_files, review_media_submission,
)


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.responses.pop(0))


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


class QualityAIMediaTests(unittest.TestCase):
    def test_gemini_is_optional_without_key(self):
        client, model = build_gemini_client({})
        self.assertIsNone(client)
        self.assertIsNone(model)

    def test_json_parser_handles_fenced_json(self):
        self.assertEqual(_parse_json('```json\n{"ok": true}\n```'), {"ok": True})

    def test_image_count_is_bounded_before_provider_call(self):
        with self.assertRaisesRegex(ValueError, "At most"):
            from quality_ai_media import inspect_images
            inspect_images(["a.png"] * (MAX_IMAGES + 1), FakeClient([]), "test-model")

    def test_video_requires_local_ffmpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.mp4"
            video.write_bytes(b"not-a-real-video")
            with patch("quality_ai_media.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "ffmpeg and ffprobe are required"):
                    extract_video_frames(video, Path(directory) / "frames")

    def test_missing_video_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                extract_video_frames(Path(directory) / "missing.mp4", directory)

    def test_media_review_rejects_unsupported_files(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.txt"
            image.write_bytes(b"not an image")
            from quality_ai_media import inspect_images
            with self.assertRaisesRegex(ValueError, "Unsupported media type"):
                inspect_images([image], FakeClient([]), "test")

    def test_media_review_bridge_preserves_advisory_governance(self):
        evidence = {
            "image_descriptions": ["A stick figure stands beside a door."],
            "visual_observations": ["The character remains visually consistent."],
            "audio_observations": ["Speech is intelligible."],
            "media_files_reviewed": {"images_or_frames": 1, "audio": True},
        }
        with patch("quality_ai_media.inspect_media_files", return_value=evidence):
            result = review_media_submission(
                {"narration": "word " * 190}, client=FakeClient([]), model="vision-test",
            )
        self.assertEqual(result["media_evidence"], evidence)
        self.assertTrue(result["review"]["governance"]["advisory_only"])
        self.assertFalse(result["review"]["governance"]["may_block_production"])


if __name__ == "__main__":
    unittest.main()
