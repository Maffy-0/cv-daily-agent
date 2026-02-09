#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import requests
from pathlib import Path

MAX_DISCORD_LEN = 1900  # 余裕を持つ

def build_github_blob_url(repo: str, sha: str, path: str) -> str:
    return f"https://github.com/{repo}/blob/{sha}/{path}"

def extract_top_items(md_text: str, top_n: int = 10) -> list[tuple[str, str]]:
    """
    Markdownから
    - "## i. TITLE"
    - "- arXiv: URL"
    を拾って (title, arxiv_url) のリストを返す
    """
    items = []
    blocks = md_text.split("\n## ")
    for b in blocks[1:]:
        lines = b.splitlines()
        title_line = lines[0].strip()  # "1. xxx"
        title = re.sub(r"^\d+\.\s*", "", title_line).strip()

        arxiv_url = ""
        for ln in lines[:15]:
            if ln.startswith("- arXiv:"):
                arxiv_url = ln.replace("- arXiv:", "").strip()
                break

        if title and arxiv_url:
            items.append((title, arxiv_url))
        if len(items) >= top_n:
            break
    return items

def post_discord(webhook_url: str, content: str) -> None:
    r = requests.post(webhook_url, json={"content": content}, timeout=20)
    r.raise_for_status()

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python notify_discord.py out/YYYY-MM-DD.md [top_n]", file=sys.stderr)
        sys.exit(2)

    md_path = Path(sys.argv[1])
    top_n = int(sys.argv[2]) if len(sys.argv) >= 3 else 10

    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        raise RuntimeError("DISCORD_WEBHOOK_URL is not set")

    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    sha = os.getenv("NEW_SHA", "").strip() or os.getenv("GITHUB_SHA", "").strip()

    md_text = md_path.read_text(encoding="utf-8")

    date_match = re.search(r"# Daily CV Digest \((\d{4}-\d{2}-\d{2})\)", md_text)
    date_str = date_match.group(1) if date_match else md_path.stem

    lines = [f"**Daily CV Digest ({date_str})**"]
    if repo and sha:
        lines.append(build_github_blob_url(repo, sha, md_path.as_posix()))

    lines.append("")
    items = extract_top_items(md_text, top_n=top_n)
    for (t, u) in items:
        lines.append(f"- {t}\n  {u}")

    content = "\n".join(lines).strip()

    if len(content) > MAX_DISCORD_LEN:
        # 超えるなら項目を削る
        while len(content) > MAX_DISCORD_LEN and items:
            items.pop()
            lines = [f"**Daily CV Digest ({date_str})**"]
            if repo and sha:
                lines.append(build_github_blob_url(repo, sha, md_path.as_posix()))
            lines.append("")
            for (t, u) in items:
                lines.append(f"- {t}\n  {u}")
            content = "\n".join(lines).strip()

    post_discord(webhook, content)

if __name__ == "__main__":
    main()
