from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ARXIV_API = "https://export.arxiv.org/api/query"


@dataclass(slots=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    abs_url: str
    pdf_url: str


def load_config(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as f:
        return tomllib.load(f)


def fetch_recent_papers(categories: list[str], max_results: int = 100) -> list[Paper]:
    query = " OR ".join(f"cat:{cat.strip()}" for cat in categories if cat.strip())
    if not query:
        raise ValueError("No arXiv categories were provided.")

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url, timeout=30) as response:
        xml_data = response.read()

    return _parse_arxiv_feed(xml_data)


def filter_last_week(papers: list[Paper]) -> list[Paper]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    return [p for p in papers if p.published >= cutoff]


def _parse_arxiv_feed(xml_data: bytes) -> list[Paper]:
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
    }
    root = ET.fromstring(xml_data)
    papers: list[Paper] = []

    for entry in root.findall("atom:entry", ns):
        abs_url = _entry_text(entry, "atom:id", ns)
        arxiv_id = abs_url.rstrip("/").split("/")[-1]
        published = datetime.fromisoformat(
            _entry_text(entry, "atom:published", ns).replace("Z", "+00:00")
        )
        title = _clean_whitespace(_entry_text(entry, "atom:title", ns))
        abstract = _clean_whitespace(_entry_text(entry, "atom:summary", ns))
        authors = [
            _clean_whitespace(author.findtext("atom:name", default="", namespaces=ns))
            for author in entry.findall("atom:author", ns)
        ]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                published=published,
                abs_url=abs_url,
                pdf_url=pdf_url,
            )
        )

    return papers


def _entry_text(entry: ET.Element, tag: str, ns: dict[str, str]) -> str:
    text = entry.findtext(tag, default="", namespaces=ns)
    if not text:
        raise ValueError(f"Missing required tag {tag} in arXiv entry")
    return text


def _clean_whitespace(text: str) -> str:
    return " ".join(text.split())


def build_scoring_prompt(papers: list[Paper], user_prompt: str, n: int) -> str:
    catalog = []
    for idx, p in enumerate(papers, start=1):
        catalog.append(
            {
                "id": str(idx),
                "title": p.title,
                "abstract": p.abstract,
            }
        )

    return (
        "You are helping rank recent arXiv papers.\n"
        f"Task: {user_prompt}\n"
        f"Return exactly the top {n} papers as a JSON array sorted by descending score.\n"
        'Each item must be: {"id": "<catalog id>", "score": <0-100 number>, "reason": "<brief>"}.\n'
        "Do not include any text outside JSON.\n"
        f"Catalog:\n{json.dumps(catalog, ensure_ascii=False)}"
    )


def score_papers(
    papers: list[Paper],
    n: int,
    user_prompt: str,
    provider: str,
    model: str,
    base_url: str,
    temperature: float = 0.1,
) -> list[tuple[Paper, float, str]]:
    prompt = build_scoring_prompt(papers, user_prompt=user_prompt, n=n)

    if provider == "ollama":
        raw = _call_ollama(
            prompt=prompt, model=model, base_url=base_url, temperature=temperature
        )
    elif provider == "llama_cpp":
        raw = _call_llama_cpp(
            prompt=prompt, model=model, base_url=base_url, temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    parsed = _extract_json_array(raw)
    scored: list[tuple[Paper, float, str]] = []

    for item in parsed:
        idx = int(item["id"]) - 1
        if idx < 0 or idx >= len(papers):
            continue
        score = float(item.get("score", 0))
        reason = str(item.get("reason", "")).strip()
        scored.append((papers[idx], score, reason))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


def _call_ollama(prompt: str, model: str, base_url: str, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    url = base_url.rstrip("/") + "/api/generate"
    response = _post_json(url, payload)
    text = response.get("response", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ollama returned an empty response.")
    return text


def _call_llama_cpp(prompt: str, model: str, base_url: str, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You respond only in valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"
    response = _post_json(url, payload)
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("llama.cpp endpoint returned no choices.")

    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("llama.cpp returned empty message content.")
    return content


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e

    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response from {url}: {body[:500]}") from e


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)

    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("Model output is not a JSON array.")
    return data


def format_apa_citation(paper: Paper) -> str:
    formatted_authors = _format_authors_apa(paper.authors)
    year = paper.published.year
    return f"{formatted_authors} ({year}). {paper.title} [Preprint]. arXiv. {paper.abs_url}"


def _format_authors_apa(authors: list[str]) -> str:
    if not authors:
        return "Unknown Author"

    formatted = []
    for name in authors:
        parts = name.split()
        if len(parts) == 1:
            formatted.append(parts[0])
            continue
        last = parts[-1]
        initials = " ".join(f"{p[0]}." for p in parts[:-1] if p)
        formatted.append(f"{last}, {initials}")

    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def download_paper_pdf(paper: Paper, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    safe_title = (
        re.sub(r"[^A-Za-z0-9._-]+", "_", paper.title).strip("_")[:80] or paper.arxiv_id
    )
    filename = f"{paper.arxiv_id}_{safe_title}.pdf"
    output = download_dir / filename

    with urllib.request.urlopen(paper.pdf_url, timeout=60) as response:
        output.write_bytes(response.read())
    return output
