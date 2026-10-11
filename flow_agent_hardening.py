"""Opt-in Google Flow Agent image-provider bridge for Relic Loop."""
from __future__ import annotations

from pathlib import Path

import pipeline as p
import flow_agent_client as flow


_INSTALLED = False


def install() -> None:
    """Make Flow Agent the first image provider when explicitly enabled.

    Normal production behavior is unchanged while FLOW_AGENT_ENABLED is off.
    When enabled, Flow Agent gets the first attempt; an unavailable/refused
    Flow Agent can fall through to the existing provider chain so a temporary
    bridge problem does not destroy an otherwise complete episode.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    if not flow.enabled():
        print("[FLOW] Integration installed but disabled (FLOW_AGENT_ENABLED=0).")
        _INSTALLED = True
        return

    original = p._generate_image_with_fallback

    def flow_first(prompt: str, out_path: Path, seed: int, references: list[Path] | None = None):
        if references:
            # The Flow Agent REST image endpoint currently accepts a prompt but
            # not the pipeline's local reference-image list. Keep continuity
            # handled by the existing chain until we wire Flow's native
            # reference-image/ingredient support in the video phase.
            print("[FLOW] Local reference supplied; using existing provider chain for this beat.")
            return original(prompt, out_path, seed, references)
        try:
            print(f"[FLOW] Google Flow image via Flow Agent: {out_path.name}")
            flow.generate_image(prompt, out_path, aspect="16:9", count=1)
            if not out_path.exists() or out_path.stat().st_size < 10000:
                raise RuntimeError("Flow Agent produced no valid image file.")
            return "flow-agent"
        except Exception as exc:
            print(f"[FLOW] Flow Agent image failed; falling back to existing providers: {exc}")
            return original(prompt, out_path, seed, references)

    p._generate_image_with_fallback = flow_first
    _INSTALLED = True
    print(f"[FLOW] Google Flow Agent is ACTIVE at {flow.FLOW_AGENT_URL}")
