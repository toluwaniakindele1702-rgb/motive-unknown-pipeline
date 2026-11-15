import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from quality_ai_media import DEFAULT_VISION_MODEL, MAX_IMAGES, cloudflare_config, extract_video_frames, inspect_audio, inspect_images, review_media_submission

class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): pass
    def json(self): return self.payload

class FakeSession:
    def __init__(self, responses): self.responses, self.calls = list(responses), []
    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs)); return FakeResponse(self.responses.pop(0))

class Tests(unittest.TestCase):
    def test_config_reads_existing_cloudflare_credentials(self):
        self.assertEqual(cloudflare_config({"CLOUDFLARE_ACCOUNT_ID":"acct","CLOUDFLARE_AUTH_TOKEN":"secret"}),
                         ("acct","secret",DEFAULT_VISION_MODEL))
    def test_missing_credentials_are_explicit(self):
        self.assertEqual(cloudflare_config({}), (None,None,DEFAULT_VISION_MODEL))
    def test_image_limit(self):
        with self.assertRaisesRegex(ValueError,"At most"):
            inspect_images(["a.png"]*(MAX_IMAGES+1), account_id="a", api_token="b")
    def test_vision_sends_image_to_cloudflare(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"frame.png"; p.write_bytes(b"imagebytes")
            session=FakeSession([{"result":{"response":'{"image_description":"A stick figure","observations":["clear"]}'}}])
            result=inspect_images([p],account_id="acct",api_token="token",session=session)
            self.assertEqual(result["image_descriptions"],["A stick figure"])
            self.assertEqual(session.calls[0][1]["headers"]["Authorization"],"Bearer token")
            self.assertIn(DEFAULT_VISION_MODEL,session.calls[0][0][0])
    def test_video_needs_ffmpeg(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"v.mp4"; p.write_bytes(b"video")
            with patch("quality_ai_media.shutil.which",return_value=None):
                with self.assertRaisesRegex(RuntimeError,"ffmpeg and ffprobe"): extract_video_frames(p,Path(d)/"f")
    def test_audio_metrics_are_local(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.wav"; p.write_bytes(b"audio")
            def run(cmd,**kwargs):
                if "ffprobe" in cmd[0]: return SimpleNamespace(returncode=0,stdout="55.0",stderr="")
                return SimpleNamespace(returncode=0,stdout="",stderr="mean_volume: -18.0 dB\nmax_volume: -1.0 dB\n")
            with patch("quality_ai_media.shutil.which",side_effect=["ffmpeg","ffprobe"]), patch("quality_ai_media.subprocess.run",side_effect=run):
                result=inspect_audio(p)
            self.assertEqual(result["duration_seconds"],55.0)
            self.assertTrue(result["not_assessed"])
    def test_advisory_bridge(self):
        with patch("quality_ai_media.inspect_media_files",return_value={"image_descriptions":["stick figure"],"media_files_reviewed":{"images_or_frames":1,"audio":False}}):
            result=review_media_submission({"narration":"word "*190},image_paths=["frame.png"],account_id="a",api_token="b")
        self.assertTrue(result["review"]["governance"]["advisory_only"])
        self.assertFalse(result["review"]["governance"]["may_block_production"])

if __name__ == "__main__": unittest.main()
