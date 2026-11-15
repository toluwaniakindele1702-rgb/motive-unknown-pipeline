import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quality_ai_media import (
    MAX_IMAGES, _parse_json, build_gemini_client, extract_video_frames,
    inspect_media_files, review_media_submission, inspect_images, inspect_audio,
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

    def test_image_adapter_builds_multimodal_parts_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"mock-png-data")
            expected = {"image_descriptions": ["A red stick figure."], "observations": []}
            with patch("quality_ai_media._generate", return_value=expected) as generate:
                result = inspect_images([image], FakeClient([]), "test-model")
            self.assertEqual(result, expected)
            self.assertEqual(len(generate.call_args.args[2]), 2)

    def test_audio_adapter_builds_multimodal_part_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "voice.wav"
            audio.write_bytes(b"mock-wave-data")
            expected = {"audio_observations": ["Speech is clear."]}
            with patch("quality_ai_media._generate", return_value=expected) as generate:
                result = inspect_audio(audio, FakeClient([]), "test-model")
            self.assertEqual(result, expected)
            self.assertEqual(len(generate.call_args.args[2]), 1)

    def test_video_frame_extraction_samples_full_duration(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.mp4"
            video.write_bytes(b"video")
            output = Path(directory) / "frames"
            def fake_run(command, **kwargs):
                if "ffprobe" in command[0]:
                    return SimpleNamespace(returncode=0, stdout="60.0", stderr="")
                Path(command[-1].replace("%02d", "01")).write_bytes(b"frame")
                self.assertIn("fps=0.13333333", command[command.index("-vf") + 1])
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch("quality_ai_media.shutil.which", side_effect=["/usr/bin/ffmpeg", "/usr/bin/ffprobe"]), patch("quality_ai_media.subprocess.run", side_effect=fake_run):
                frames = extract_video_frames(video, output, max_frames=8)
            self.assertEqual(len(frames), 1)
            self.assertTrue(frames[0].is_file())

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
