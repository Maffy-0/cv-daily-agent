#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Daily CV Digest (config.yaml driven)
- Fetch arXiv papers by query
- Deduplicate using seen.json
- Filter by keywords (optional)
- Summarize with Gemini (optional, supports batch)
- Write out/YYYY-MM-DD.md

Usage:
  python daily_cv_digest.py config.yaml

Env:
  GOOGLE_API_KEY (recommended via GitHub Actions Secrets)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from zoneinfo import ZoneInfo

import arxiv
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()


# -----------------------
# Config / IO helpers
# -----------------------

def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def load_seen(seen_path: Path) -> Dict[str, str]:
    if not seen_path.exists():
        return {}
    try:
        return json.loads(seen_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_seen(seen_path: Path, seen: Dict[str, str]) -> None:
    seen_path.write_text(json.dumps(seen, ensure_ascii=False, indent=2), encoding="utf-8")


def today_str_jst() -> str:
    return dt.datetime.now(ZoneInfo("Asia/Tokyo")).date().isoformat()


# -----------------------
# arXiv fetch
# -----------------------

def fetch_arxiv(query: str, max_results: int, delay_seconds: float = 3.0) -> List[dict]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client(
        page_size=min(max_results, 100),
        delay_seconds=delay_seconds,
        num_retries=3,
    )

    results: List[dict] = []
    for r in client.results(search):
        results.append(
            {
                "id": r.get_short_id(),
                "title": (r.title or "").strip().replace("\n", " "),
                "authors": [a.name for a in (r.authors or [])],
                "published": r.published.isoformat() if r.published else "",
                "updated": r.updated.isoformat() if r.updated else "",
                "summary": (r.summary or "").strip().replace("\n", " "),
                "pdf_url": r.pdf_url,
                "abs_url": r.entry_id,
            }
        )
    return results


# -----------------------
# Filtering
# -----------------------

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def passes_gate(title: str, abstract: str, required_any: List[str]) -> bool:
    if not required_any:
        return True
    text = f"{title}\n{abstract}".lower()
    return any((w.strip().lower() in text) for w in required_any if w.strip())


def keyword_score(title: str, abstract: str, keywords: List[str], exclude: List[str]) -> Tuple[int, List[str]]:
    text = f"{title}\n{abstract}".lower()

    for ex in exclude:
        ex2 = ex.strip().lower()
        if ex2 and ex2 in text:
            return -999, [f"excluded:{ex.strip()}"]

    hits: List[str] = []
    score = 0
    t_low = (title or "").lower()

    for kw in keywords:
        k = kw.strip()
        if not k:
            continue
        k_low = k.lower()
        if k_low in text:
            hits.append(k)
            score += 3 if k_low in t_low else 1

    return score, hits


# -----------------------
# Gemini
# -----------------------

def gemini_generate(
    model: str,
    api_key: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }

    last_debug = ""
    for attempt in range(5):
        resp = requests.post(url, json=payload, timeout=60)

        # retry on transient errors
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue

        try:
            resp.raise_for_status()
        except Exception as e:
            body = resp.text[:500].replace("\n", " ")
            raise RuntimeError(f"HTTP {resp.status_code}: {e}; body={body}")

        data = resp.json() if resp.content else {}

        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            return f"（要約生成に失敗: API error code={err.get('code')} status={err.get('status')} msg={err.get('message')}）"

        pf = data.get("promptFeedback") or {}
        block = pf.get("blockReason")
        safety = pf.get("safetyRatings")

        candidates = data.get("candidates") or []
        if not candidates:
            last_debug = f"no candidates; blockReason={block}; safetyRatings={safety}"
            time.sleep(1.5 * (attempt + 1))
            continue

        c0 = candidates[0]
        finish = c0.get("finishReason")
        content = c0.get("content") or {}
        parts = content.get("parts") or []

        texts = []
        for p in parts:
            t = p.get("text")
            if t:
                texts.append(t)
        text_out = "\n".join(texts).strip()

        if text_out:
            return text_out

        last_debug = f"empty text; finishReason={finish}; parts={len(parts)}; blockReason={block}"
        time.sleep(1.5 * (attempt + 1))

    return f"（要約生成に失敗: retry exhausted; last={last_debug}）"


def summarize_with_gemini(model: str, title: str, abstract: str, max_output_tokens: int, temperature: float) -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    prompt = f"""
あなたはコンピュータビジョン研究のキュレーターです。
以下の論文を日本語で短く要約してください。

出力フォーマット（厳守）：
- What: 何をした？
- Novelty: 新規性は？
- Why it matters: 何に効く？

Title: {title}
Abstract: {abstract}
""".strip()

    return gemini_generate(
        model=model,
        api_key=api_key,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    # Remove ```yaml ... ``` or ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def summarize_batch_with_gemini(
    model: str,
    papers: List[dict],
    max_output_tokens: int,
    temperature: float,
    abstract_max_chars: int = 1200,
) -> Dict[str, str]:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    items: List[str] = []
    for p in papers:
        pid = p.get("id", "")
        title = normalize_text(p.get("title", ""))
        abst = normalize_text(p.get("summary", ""))[:abstract_max_chars]
        items.append(
            f"- id: {pid}\n"
            f"  title: {title}\n"
            f"  abstract: {abst}\n"
        )

    prompt = f"""
あなたはコンピュータビジョン研究のキュレーターです。
以下の各論文について日本語で短く要約してください。

必ず次の YAML だけを出力してください（前置きや説明文は禁止）：
summaries:
  "<id>": |
    - What: 何をした？
    - Novelty: 新規性は？
    - Why it matters: 何に効く？

注意:
- id は入力の id をそのまま使う
- summaries 以外のキーは出力しない
- YAML を壊さない（インデント厳守）
- コードフェンス（```）を付けない

論文一覧:
{chr(10).join(items)}
""".strip()

    text = gemini_generate(
        model=model,
        api_key=api_key,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    text = _strip_code_fences(text)

    try:
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            summaries = data.get("summaries", {})
            if isinstance(summaries, dict):
                out: Dict[str, str] = {}
                for k, v in summaries.items():
                    if isinstance(k, str) and isinstance(v, str):
                        out[k.strip()] = v.strip()
                return out
    except Exception:
        pass

    return {}


# -----------------------
# Markdown writer
# -----------------------

def write_markdown(date_str: str, items: List[dict], out_path: Path) -> None:
    lines: List[str] = []
    lines.append(f"# Daily CV Digest ({date_str})")
    lines.append("")
    lines.append(f"- Total: {len(items)}")
    lines.append("")

    for i, it in enumerate(items, start=1):
        title = normalize_text(it.get("title", ""))
        abs_url = it.get("abs_url", "")
        pdf_url = it.get("pdf_url", "")
        authors_list = it.get("authors", [])
        authors = ", ".join(authors_list[:8]) + ("..." if len(authors_list) > 8 else "")
        abstract = normalize_text(it.get("summary", ""))
        llm_sum = (it.get("llm_summary") or "").strip()
        kw_score = it.get("kw_score", None)
        kw_hits = it.get("kw_hits", [])

        lines.append(f"## {i}. {title}")
        lines.append(f"- arXiv: {abs_url}")
        lines.append(f"- PDF: {pdf_url}")
        lines.append(f"- Authors: {authors}")
        if kw_score is not None:
            hits_str = ", ".join(kw_hits[:12])
            lines.append(f"- Keyword score: {kw_score} / hits: {hits_str}")
        lines.append("")
        lines.append("<details><summary>Abstract</summary>")
        lines.append("")
        lines.append(abstract)
        lines.append("")
        lines.append("</details>")
        lines.append("")
        if llm_sum:
            lines.append("**LLM Summary**")
            lines.append("")
            lines.append(llm_sum)
            lines.append("")

    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


# -----------------------
# Main
# -----------------------

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python daily_cv_digest.py config.yaml", file=sys.stderr)
        sys.exit(2)

    cfg = load_config(sys.argv[1])

    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data"))
    out_dir = Path(cfg.get("paths", {}).get("out_dir", "out"))
    seen_path = Path(cfg.get("paths", {}).get("seen_path", str(data_dir / "seen.json")))

    ensure_dirs(data_dir, out_dir)

    date_str = today_str_jst()

    arxiv_cfg = cfg.get("arxiv", {})
    query = arxiv_cfg.get("query", "cat:cs.CV")
    max_results = int(arxiv_cfg.get("max_results", 80))
    delay_seconds = float(arxiv_cfg.get("delay_seconds", 3.0))
    only_new = bool(arxiv_cfg.get("only_new", True))

    papers = fetch_arxiv(query=query, max_results=max_results, delay_seconds=delay_seconds)

    seen = load_seen(seen_path)
    fresh = [p for p in papers if p["id"] not in seen]
    chosen = fresh if only_new else papers

    # keyword filter (optional)
    filt_cfg = cfg.get("filter", {})
    enable_filter = bool(filt_cfg.get("enable", True))
    if enable_filter:
        keywords = filt_cfg.get("keywords", []) or []
        exclude = filt_cfg.get("exclude", []) or []
        min_score = int(filt_cfg.get("min_score", 2))
        required_any = filt_cfg.get("required_any", []) or []
        top_k = int(filt_cfg.get("top_k", 0))

        filtered: List[dict] = []
        for p in chosen:
            title = p.get("title", "")
            abst = p.get("summary", "")

            if not passes_gate(title, abst, required_any):
                continue

            sc, hits = keyword_score(title, abst, keywords, exclude)
            if sc >= min_score:
                p["kw_score"] = sc
                p["kw_hits"] = hits
                filtered.append(p)

        filtered.sort(key=lambda x: x.get("kw_score", 0), reverse=True)
        if top_k > 0:
            filtered = filtered[:top_k]
    else:
        filtered = chosen

    # summarizer (optional)
    sum_cfg = cfg.get("summarizer", {})
    enable_sum = bool(sum_cfg.get("enable", False))
    if enable_sum and filtered:
        model = str(sum_cfg.get("model", "gemini-2.5-flash-lite"))
        temperature = float(sum_cfg.get("temperature", 0.3))
        per_paper_tokens = int(sum_cfg.get("max_output_tokens", 180))

        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            print("[gemini] GOOGLE_API_KEY not set", file=sys.stderr)
        else:
            # --- sanity check once ---
            try:
                test = gemini_generate(
                    model=model,
                    api_key=api_key,
                    prompt="Say 'OK' only.",
                    max_output_tokens=8,
                    temperature=0.0,
                )
                print(f"[gemini sanity] {test[:80]}")
            except Exception as e:
                print(f"[gemini sanity error] {e}")

            # batch config
            batch_cfg = (sum_cfg.get("batch", {}) or {})
            batch_enable = bool(batch_cfg.get("enable", True))
            batch_size = int(batch_cfg.get("batch_size", 5))
            abstract_max_chars = int(batch_cfg.get("abstract_max_chars", 1200))
            fallback_single = bool(batch_cfg.get("fallback_single_on_missing", True))

            # safety clamp
            batch_size = max(batch_size, 1)
            abstract_max_chars = max(abstract_max_chars, 200)

            # batch max output tokens (important!)
            batch_max_tokens = batch_cfg.get("max_output_tokens", None)
            if batch_max_tokens is None:
                batch_max_tokens = per_paper_tokens * batch_size
            batch_max_tokens = int(batch_max_tokens)

            if batch_enable and batch_size > 1:
                for i in range(0, len(filtered), batch_size):
                    chunk = filtered[i:i + batch_size]
                    try:
                        mapping = summarize_batch_with_gemini(
                            model=model,
                            papers=chunk,
                            max_output_tokens=batch_max_tokens,
                            temperature=temperature,
                            abstract_max_chars=abstract_max_chars,
                        )
                        for p in chunk:
                            pid = p["id"]
                            if pid in mapping:
                                p["llm_summary"] = mapping[pid]
                            else:
                                p["llm_summary"] = "（要約生成に失敗: batch missing id）"

                        # optional fallback for missing ones
                        if fallback_single:
                            missing = [p for p in chunk if "batch missing id" in (p.get("llm_summary") or "")]
                            for p in missing:
                                try:
                                    p["llm_summary"] = summarize_with_gemini(
                                        model=model,
                                        title=p.get("title", ""),
                                        abstract=p.get("summary", ""),
                                        max_output_tokens=per_paper_tokens,
                                        temperature=temperature,
                                    )
                                except Exception as e:
                                    p["llm_summary"] = f"（要約エラー: {e}）"

                    except Exception as e:
                        for p in chunk:
                            p["llm_summary"] = f"（要約エラー: {e}）"
            else:
                for p in filtered:
                    try:
                        p["llm_summary"] = summarize_with_gemini(
                            model=model,
                            title=p.get("title", ""),
                            abstract=p.get("summary", ""),
                            max_output_tokens=per_paper_tokens,
                            temperature=temperature,
                        )
                    except Exception as e:
                        p["llm_summary"] = f"（要約エラー: {e}）"

    # update seen (mark all fetched, not just filtered)
    for p in papers:
        seen[p["id"]] = date_str
    save_seen(seen_path, seen)

    out_path = out_dir / f"{date_str}.md"
    write_markdown(date_str, filtered, out_path)

    print(f"✅ Wrote: {out_path} (kept={len(filtered)}, fetched={len(papers)}, new={len(fresh)})")


if __name__ == "__main__":
    main()
