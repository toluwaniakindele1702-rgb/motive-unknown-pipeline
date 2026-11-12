"""One-time local helper to export a user-authenticated Raphael Playwright session.

Run locally. This opens a visible browser; the user signs in and completes any
human verification themselves. No credentials are collected by this script.
The resulting storage-state file is sensitive and must never be committed.
"""
from pathlib import Path
import os

from playwright.sync_api import sync_playwright

RAPHAEL_URL = "https://raphael.app/gpt-image-2-6-sunburst"
OUTPUT = Path.home() / "raphael_storage_state.json"


def main() -> None:
    print("A visible Chromium window will open.")
    print("Sign in to Raphael in that window and complete any prompts yourself.")
    print("When you can see your signed-in Raphael account, return here.")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        try:
            page.goto(RAPHAEL_URL, wait_until="domcontentloaded", timeout=90000)
            input("Press Enter here only after you have successfully signed in: ")
            body = (page.locator("body").inner_text(timeout=15000) or "").lower()
            if any(term in body for term in ("sign in to continue", "sign in required", "log in to generate")):
                raise RuntimeError("Raphael still appears signed out. No session file was saved.")
            context.storage_state(path=str(OUTPUT))
            try:
                os.chmod(OUTPUT, 0o600)
            except OSError:
                pass
            print(f"Session state saved locally to: {OUTPUT}")
            print("Do not share this file or commit it to Git.")
            print("Next, add it to GitHub Actions secrets using the GitHub CLI.")
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    main()
