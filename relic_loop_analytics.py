#!/usr/bin/env python3
"""Collect post-publication YouTube performance and retention metrics when OAuth permits."""
from __future__ import annotations
import json, os, sys
from datetime import date, timedelta
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN=os.getenv("YOUTUBE_TOKEN_JSON","")
CLIENT=os.getenv("YOUTUBE_CLIENT_SECRET_JSON","")
UPLOAD=Path("youtube_public_upload.json")
OUT=Path("relic_loop_output/analytics_latest.json")
ANALYTICS_SCOPE="https://www.googleapis.com/auth/yt-analytics.readonly"

def main():
    if not TOKEN or not CLIENT or not UPLOAD.is_file():
        print("Analytics skipped: OAuth or successful upload record is unavailable.")
        return 0
    token=json.loads(TOKEN); client=json.loads(CLIENT)
    section=client.get("installed") or client.get("web") or client
    if isinstance(section,dict):
        token.setdefault("client_id",section.get("client_id"))
        token.setdefault("client_secret",section.get("client_secret"))
    creds=Credentials.from_authorized_user_info(token,scopes=[
        "https://www.googleapis.com/auth/youtube.readonly", ANALYTICS_SCOPE])
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("Analytics skipped: YouTube OAuth cannot be refreshed.")
            return 0
    if not creds.has_scopes([ANALYTICS_SCOPE]):
        result={"status":"requires_reauthorization","reason":"The saved YouTube OAuth grant lacks yt-analytics.readonly.",
                "needed_scope":ANALYTICS_SCOPE,
                "metrics_not_available":["audience retention","average view duration","average view percentage",
                                         "viewed-versus-swiped-away","subscribers gained"]}
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
        print("Analytics needs a one-time YouTube OAuth reauthorization with yt-analytics.readonly; upload remains successful.")
        return 0
    upload=json.loads(UPLOAD.read_text(encoding="utf-8"))
    video_id=upload.get("video_id")
    if not video_id:
        print("Analytics skipped: upload record contains no video ID.")
        return 0
    end=date.today()-timedelta(days=1)
    start=end-timedelta(days=27)
    service=build("youtubeAnalytics","v2",credentials=creds,cache_discovery=False)
    metrics="views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,likes,comments,subscribersGained"
    report=service.reports().query(ids="channel==MINE",startDate=start.isoformat(),endDate=end.isoformat(),
        metrics=metrics,dimensions="day",filters=f"video=={video_id}",sort="day").execute()
    result={"status":"collected","video_id":video_id,"video_url":upload.get("url"),
            "date_range":{"start":start.isoformat(),"end":end.isoformat()},
            "column_headers":report.get("columnHeaders",[]),"rows":report.get("rows",[]),
            "note":"Viewed-versus-swiped-away is not exposed by this standard Analytics query; use the Shorts-specific Studio metric where available."}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"video_id":video_id,"days":len(result["rows"])}))
    return 0

if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps({"status":"failed","error":f"{type(exc).__name__}: {exc}"},indent=2),encoding="utf-8")
        print(f"Analytics collection failed without affecting the successful video upload: {type(exc).__name__}: {exc}")
        raise SystemExit(0)
