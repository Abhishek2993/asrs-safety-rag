"""Capture README screenshots of a running Streamlit app with Playwright.

Usage (with the app running, e.g. ``streamlit run app/app.py --server.headless true``)::

    pip install -e ".[screenshots]"
    python scripts/screenshot_app.py --url http://localhost:8501
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"


def wait_for_app(page, text: str) -> None:
    """Wait until Streamlit has rendered ``text`` and finished running the script."""
    page.wait_for_selector("[data-testid='stApp']", timeout=120_000)
    page.get_by_text(text).first.wait_for(timeout=300_000)
    page.wait_for_function(
        "() => !document.querySelector(\"[data-testid='stStatusWidget']\")", timeout=300_000
    )
    page.wait_for_timeout(1500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://localhost:8501")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    executable = os.environ.get("CHROMIUM_PATH")
    with sync_playwright() as p:
        browser = (
            p.chromium.launch(executable_path=executable) if executable else (p.chromium.launch())
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1300}, device_scale_factor=2)
        page.goto(args.url)
        wait_for_app(page, "Predicted contributing factors")
        page.wait_for_selector("[data-testid='stVegaLiteChart']", timeout=120_000)
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "app_query.png"), full_page=False)
        page.get_by_role("tab", name="Trends").click()
        wait_for_app(page, "Trends across reports")
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "app_trends.png"), full_page=False)
        browser.close()
    print(f"Saved screenshots to {OUT}")


if __name__ == "__main__":
    main()
