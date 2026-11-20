#!/usr/bin/env python3
"""Publish a Relic Loop Short publicly; fail closed on missing/invalid OAuth."""
import json, os, sys
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
ROOT=Path(__file__).resolve().parent
VIDEO=Path(os.environ.get("RELIC_LOOP_VIDEO_PATH","")).resolve()
TOKEN=os.getenv("YOUTUBE_TOKEN_JSON","")
CLIENT=os.getenv("YOUTUBE_CLIENT_SECRET_JSON","")
if not TOKEN or not CLIENT: raise SystemExit("Missing YOUTUBE_TOKEN_JSON or YOUTUBE_CLIENT_SECRET_JSON secrets")
token=json.loads(TOKEN); client_info=json.loads(CLIENT)
client_section=client_info.get("installed") or client_info.get("web") or client_info
if isinstance(client_section,dict):
    token.setdefault("client_id",client_section.get("client_id"))
    token.setdefault("client_secret",client_section.get("client_secret"))
creds=Credentials.from_authorized_user_info(token,scopes=["https://www.googleapis.com/auth/youtube.upload","https://www.googleapis.com/auth/youtube.readonly","https://www.googleapis.com/auth/youtube.force-ssl"])
if not creds.valid:
    if creds.expired and creds.refresh_token: creds.refresh(Request())
    else: raise SystemExit("YouTube OAuth invalid and not refreshable")
service=build("youtube","v3",credentials=creds,cache_discovery=False)
channels=service.channels().list(part="id",mine=True).execute().get("items",[])
if not channels: raise SystemExit("OAuth cannot access a YouTube channel")
if "--check-auth" in sys.argv:
    print("YOUTUBE_AUTH_PREFLIGHT=success")
    raise SystemExit(0)
if not VIDEO.is_file(): raise SystemExit("MP4 path missing")
report_path=Path(os.environ.get("RELIC_LOOP_REPORT_PATH","run_report.json"))
report=json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
title=str(report.get("story",{}).get("title","Relic Loop Short"))[:95]
description="An original short story from Relic Loop. #shorts #animation #comedy"
body={"snippet":{"title":title,"description":description,"categoryId":"23"},"status":{"privacyStatus":"public","selfDeclaredMadeForKids":False}}
request=service.videos().insert(part="snippet,status",body=body,media_body=MediaFileUpload(str(VIDEO),mimetype="video/mp4",chunksize=8*1024*1024,resumable=True))
response=None
while response is None:
    status,response=request.next_chunk()
vid=response.get("id")
if not vid: raise SystemExit("YouTube upload response had no video ID")
result={"ok":True,"video_id":vid,"url":"https://www.youtube.com/watch?v="+vid,"privacy_status":"public","title":title}
Path("youtube_public_upload.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
# Optional sidecar subtitle upload; never invalidate a successful public upload.
caption_path = Path(os.environ.get("RELIC_LOOP_CAPTIONS_PATH", ""))
if caption_path.is_file():
    try:
        caption_request = service.captions().insert(
            part="snippet",
            body={"snippet": {"videoId": vid, "language": "en", "name": "English word-by-word captions",
                              "isDraft": False}},
            media_body=MediaFileUpload(str(caption_path), mimetype="application/octet-stream", resumable=True))
        caption_response = caption_request.execute()
        result["captions"] = {"uploaded": True, "caption_id": caption_response.get("id"),
                              "format": "SRT", "word_level": True}
    except Exception as caption_error:
        result["captions"] = {"uploaded": False,
                              "reason": f"{type(caption_error).__name__}: {str(caption_error)[:400]}",
                              "sidecar_available": True}
    Path("youtube_public_upload.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print("PUBLIC_UPLOAD_URL="+result["url"])
print("CAPTION_UPLOAD="+json.dumps(result.get("captions", {"uploaded": False, "reason": "No SRT path supplied"})))
