from __future__ import annotations

import argparse
import os
from pathlib import Path

from .core import (
    download_paper_pdf,
    fetch_recent_papers,
    filter_last_week,
    format_apa_citation,
    load_config,
    score_papers,
)


def get_default_config_path() -> Path:
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base_dir = Path(xdg_config_home).expanduser() if xdg_config_home else Path.home() / ".config"
    return base_dir / "paper_picker.toml"


DEFAULT_CONFIG_PATH = get_default_config_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper Picker: rank recent arXiv papers with local LLMs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=get_default_config_path(),
        help="Path to config TOML file",
    )
    parser.add_argument(
        "-n",
        "--num",
        dest="n",
        type=int,
        required=True,
        help="Number of top papers to return",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    categories = config.get("arxiv", {}).get("categories", [])
    max_results = int(config.get("arxiv", {}).get("max_results", 100))

    llm_cfg = config.get("llm", {})
    provider = str(llm_cfg.get("provider", "ollama"))
    model = str(llm_cfg.get("model", ""))
    base_url = str(llm_cfg.get("base_url", "http://localhost:11434"))
    temperature = float(llm_cfg.get("temperature", 0.1))
    user_prompt = str(
        llm_cfg.get(
            "prompt",
            "Select papers that are novel, technically sound, and relevant to my research interests.",
        )
    )

    print(f"Fetching arXiv feed for categories: {', '.join(categories)}")
    papers = fetch_recent_papers(categories=categories, max_results=max_results)
    week_papers = filter_last_week(papers)

    if not week_papers:
        print("No papers found in the last week for these categories.")
        return

    print(
        f"Found {len(week_papers)} papers from the last week. Scoring with {provider}/{model}..."
    )
    ranked = score_papers(
        week_papers,
        n=args.n,
        user_prompt=user_prompt,
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
    )

    if not ranked:
        print("Model returned no ranked papers.")
        return

    print("\nTop papers:\n")
    for idx, (paper, score, reason) in enumerate(ranked, start=1):
        print(f"[{idx}] Score {score:.1f} | {paper.title}")
        print(f"    arXiv: {paper.abs_url}")
        print(f"    Reason: {reason or '(no reason provided)'}")
        print()

    selected = (
        input("Select papers to download (e.g. 1,3 or 'all'; blank to skip): ")
        .strip()
        .lower()
    )
    if not selected:
        print("No downloads selected.")
        return

    chosen_indices = _parse_selection(selected, len(ranked))
    if not chosen_indices:
        print("No valid selections; exiting.")
        return

    downloads_dir = Path.home() / "Downloads"
    print(f"Downloading to {downloads_dir} ...")

    for idx in chosen_indices:
        paper, _, _ = ranked[idx]
        path = download_paper_pdf(paper, downloads_dir)
        print(f"Downloaded: {path}")
        print(f"APA: {format_apa_citation(paper)}")


def _parse_selection(raw: str, count: int) -> list[int]:
    if raw == "all":
        return list(range(count))

    result: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part.isdigit():
            continue
        idx = int(part)
        if 1 <= idx <= count:
            result.add(idx - 1)
    return sorted(result)


if __name__ == "__main__":
    main()
