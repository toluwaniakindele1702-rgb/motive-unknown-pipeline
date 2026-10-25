#!/usr/bin/env python3
"""Upload the finished Chispa-Pifia Short with the existing YouTube OAuth setup."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "state" / "current_run" / "relic_loop_short.mp4"
OUTPUT_PATH = ROOT / "state" / "current_run" / "youtube_upload.json"
TOKEN_JSON = os.environ.get("YOUTUBE_TOKEN_JSON", "").strip()
CLIENT_SECRET_JSON = os.environ.get("YOUTUBE_CLIENT_SECRET_JSON", "").strip()
PRIVACY = os.environ.get("YOUTUBE_PRIVACY_STATUS", "private").strip().lower()
CHANNEL_NAME = os.environ.get("CHANNEL_NAME", "Chispa-Pifia").strip() or "Chispa-Pifia"
SCOPE = "https://www.googleapis.com/auth/youtube.upload"

if PRIVACY not in {"public", "private", "unlisted"}:
    PRIVACY = "private"


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_value(data: Any, keys: set[str]) -> str | None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in keys and isinstance(value, str) and value.strip():
                return value.strip()
        for value in data.values():
            found = find_value(value, keys)
            if found:
                return found
    elif isinstance(data, list):
        for value in data:
            found = find_value(value, keys)
            if found:
                return found
    return None


def metadata() -> tuple[str, str]:
    candidates = [
        ROOT / "state" / "current_run" / "flow_manifest.json",
        ROOT / "state" / "current_run" / "relic_loop_short.json",
        ROOT / "state" / "current_run" / "story.json",
    ]
    merged: list[Any] = []
    for path in candidates:
        data = load_json(path)
        if data is not None:
            merged.append(data)

    title = None
    description = None
    for data in merged:
        title = title or find_value(data, {"title", "video_title", "youtube_title"})
        description = description or find_value(data, {"description", "video_description", "youtube_description"})

    if not title:
        title = "The Worst Decision He Could Have Made 😭"
    if not description:
        description = (
            "Ridiculous stories. Terrible decisions. Somehow, things always get worse. "
            "Welcome to Chispa-Pifia. ⚡\n\n"
            "#shorts #comedy #stickman"
        )

    # Keep titles comfortably below YouTube's title limit.
    title = title[:95].strip()
    return title, description


def credentials() -> Credentials:
    if not TOKEN_JSON or not CLIENT_SECRET_JSON:
        raise RuntimeError("YOUTUBE_TOKEN_JSON and YOUTUBE_CLIENT_SECRET_JSON are required")

    token_info = json.loads(TOKEN_JSON)
    client_info = json.loads(CLIENT_SECRET_JSON)
    if not isinstance(token_info, dict) or not isinstance(client_info, dict):
        raise RuntimeError("YouTube OAuth secrets must contain JSON objects")

    creds = Credentials.from_authorized_user_info(token_info, scopes=[SCOPE])
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError("Stored YouTube OAuth credentials are invalid and cannot be refreshed")
    return creds


def upload() -> dict[str, Any]:
    if not VIDEO_PATH.exists() or VIDEO_PATH.stat().st_size == 0:
        raise RuntimeError(f"Finished Short not found: {VIDEO_PATH}")

    creds = credentials()
    service = build("youtube", "v3", credentials=creds, cache_discovery=False)
    title, description = metadata()

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": "23",  # Comedy
        },
        "status": {
            "privacyStatus": PRIVACY,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(VIDEO_PATH),
        mimetype="video/mp4",
        chunksize=8 * 1024 * 1024,
        resumable=True,
    )
    request = service.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            while response is None:
                status, response = request.next_chunk()
                if status:
                    print(f"[YOUTUBE] upload progress: {status.progress() * 100:.1f}%")
            break
        except HttpError as exc:
            last_error = exc
            if exc.resp.status not in {500, 502, 503, 504} or attempt == 5:
                raise
            delay = min(60, 2 ** attempt)
            print(f"[YOUTUBE] transient HTTP {exc.resp.status}; retrying in {delay}s")
            time.sleep(delay)
        except OSError as exc:
            last_error = exc
            if attempt == 5:
                raise
            delay = min(60, 2 ** attempt)
            print(f"[YOUTUBE] transient upload error; retrying in {delay}s: {exc}")
            time.sleep(delay)

    if not response or not response.get("id"):
        raise RuntimeError(f"YouTube upload returned no video ID: {last_error}")

    video_id = response["id"]
    result = {
        "ok": True,
        "channel_name": CHANNEL_NAME,
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "privacy_status": PRIVACY,
        "title": title,
        "uploaded_file": str(VIDEO_PATH),
    }
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[YOUTUBE] Uploaded {CHANNEL_NAME}: {result['url']}")
    return result


if __name__ == "__main__":
    result = upload()
    print(json.dumps(result, ensure_ascii=False, indent=2))
