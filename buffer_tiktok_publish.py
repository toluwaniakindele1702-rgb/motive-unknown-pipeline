#!/usr/bin/env python3
"""Queue a finished Relic Loop MP4 for automatic TikTok publishing via Buffer.

Buffer fetches media from a public URL, so the workflow creates a public GitHub
Release asset first. The Buffer API key is read only from BUFFER_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_URL = "https://api.buffer.com"


class BufferPublishError(RuntimeError):
    pass


def _graphql(query: str, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        API_URL,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise BufferPublishError(f"Buffer API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise BufferPublishError(f"Could not reach Buffer API: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise BufferPublishError("Buffer API returned an invalid response")
    if payload.get("errors"):
        messages = [str(item.get("message", "GraphQL error")) for item in payload["errors"]]
        raise BufferPublishError("Buffer API GraphQL error: " + "; ".join(messages)[:1200])
    data = payload.get("data")
    if not isinstance(data, dict):
        raise BufferPublishError("Buffer API response did not contain data")
    return data


def _gql_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def discover_tiktok_channel(api_key: str, configured_channel_id: str = "") -> str:
    if configured_channel_id.strip():
        return configured_channel_id.strip()

    account = _graphql(
        "query { account { organizations { id name } } }",
        api_key,
    )
    organizations = (account.get("account") or {}).get("organizations") or []
    matches: list[str] = []
    for organization in organizations:
        organization_id = str(organization.get("id") or "").strip()
        if not organization_id:
            continue
        channels_data = _graphql(
            "query { channels(input: { organizationId: "
            + _gql_string(organization_id)
            + " }) { id name displayName service isQueuePaused } }",
            api_key,
        )
        for channel in channels_data.get("channels") or []:
            service = str(channel.get("service") or "").lower()
            if service == "tiktok" or "tiktok" in service:
                channel_id = str(channel.get("id") or "").strip()
                if channel_id:
                    matches.append(channel_id)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise BufferPublishError(
            "No TikTok channel was found in this Buffer account. "
            "Connect the TikTok profile in Buffer, then retry."
        )
    raise BufferPublishError(
        "More than one TikTok channel is connected. Set BUFFER_TIKTOK_CHANNEL_ID "
        "to the channel ID you want Nexus to publish to."
    )


def build_caption(report: dict[str, Any]) -> str:
    story = report.get("story") or {}
    title = str(story.get("title") or "A Relic Loop short story").strip()
    caption = title[:115].rstrip() + " #storytime #animation"
    return caption[:150]


def publish_video(video_url: str, report_path: str | Path) -> dict[str, Any]:
    api_key = (os.getenv("BUFFER_API_KEY") or "").strip()
    if not api_key:
        raise BufferPublishError("BUFFER_API_KEY GitHub Actions secret is missing")
    channel_id = discover_tiktok_channel(
        api_key, os.getenv("BUFFER_TIKTOK_CHANNEL_ID", "")
    )
    report_file = Path(report_path)
    report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.is_file() else {}
    caption = build_caption(report)

    mutation = (
        "mutation { createPost(input: { "
        "text: " + _gql_string(caption) + ", "
        "channelId: " + _gql_string(channel_id) + ", "
        "schedulingType: automatic, mode: addToQueue, "
        "assets: [{ video: { url: " + _gql_string(video_url)
        + ", metadata: { thumbnailOffset: 1000 } } }] "
        "}) { ... on PostActionSuccess { post { id text dueAt } } "
        "... on MutationError { message } } }"
    )
    result = _graphql(mutation, api_key).get("createPost") or {}
    if result.get("message"):
        raise BufferPublishError(str(result["message"]))
    post = result.get("post")
    if not isinstance(post, dict) or not post.get("id"):
        raise BufferPublishError(
            "Buffer did not confirm creation of the TikTok post: "
            + json.dumps(result, ensure_ascii=False)[:1000]
        )
    return {
        "ok": True,
        "platform": "tiktok",
        "provider": "buffer",
        "buffer_post_id": post["id"],
        "caption": caption,
        "due_at": post.get("dueAt"),
        "video_url": video_url,
        "publishing_mode": "automatic",
        "queue_mode": "addToQueue",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Queue a Relic Loop video in Buffer")
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", default="buffer_tiktok_result.json")
    args = parser.parse_args()
    try:
        result = publish_video(args.video_url, args.report)
    except Exception as exc:
        print(f"BUFFER_TIKTOK_PUBLISH_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("BUFFER_TIKTOK_PUBLISH_SUCCESS")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
