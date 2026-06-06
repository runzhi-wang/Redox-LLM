"""Answer citation cleanup and source link helpers."""

from __future__ import annotations

import html
import re

# [10.1016/xxx #Section title] or [doi #section]
_INLINE_CITE_RE = re.compile(
    r"\[10\.\S+ ?#[^\]]+\]",
    re.IGNORECASE,
)
# leftover [10.xxx/...] without #
_INLINE_DOI_RE = re.compile(r"\[(10\.\S+)\]")


def section_label(ref: dict) -> str:
    section = (ref.get("section") or "").strip()
    if section:
        return section
    path = (ref.get("heading_path") or "").strip()
    if path:
        return path.split(" > ")[-1].strip()
    return "正文片段"


def strip_inline_citations(answer: str) -> str:
    """Remove verbose [doi #section] blocks from model answer body."""
    text = _INLINE_CITE_RE.sub("", answer)
    text = _INLINE_DOI_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_PREAMBLE_LINE_RE = re.compile(
    r"^(?:"
    r"简要回答[（(][^）)\n]*[）)]?\s*[：:]"
    r"|简短结论\s*[：:]"
    r"|下面归纳文献片段[^：:\n]*[：:]"
    r"|基于你提供的文献片段[^：:\n]*[：:]"
    r"|根据提供的文献片段[^：:\n]*[：:]"
    r"|仅使用片段信息[^：:\n]*[：:]"
    r"|以下回答基于[^：:\n]*[：:]"
    r")\s*$",
    re.IGNORECASE,
)


def strip_answer_preamble(answer: str) -> str:
    """Remove meta prefaces like '简要回答（基于你提供的文献片段）：'."""
    text = answer.strip()
    text = re.sub(
        r"^简要回答[（(][^）)\n]*[）)]?\s*[：:]\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^简短结论\s*[：:]\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    lines = text.splitlines()
    while lines and _PREAMBLE_LINE_RE.match(lines[0].strip()):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def chunk_preview(ref: dict, chunks: list[dict], limit: int = 480) -> str:
    chunk_id = ref.get("chunk_id")
    for chunk in chunks:
        if chunk_id and chunk.get("chunk_id") == chunk_id:
            return (chunk.get("text") or "")[:limit]
    doi = ref.get("doi", "")
    section = ref.get("section", "")
    for chunk in chunks:
        if chunk.get("doi") == doi and chunk.get("section") == section:
            return (chunk.get("text") or "")[:limit]
    for chunk in chunks:
        if chunk.get("doi") == doi:
            return (chunk.get("text") or "")[:limit]
    return ""


def source_chip_html(ref: dict, chunks: list[dict], index: int) -> str:
    doi = html.escape(ref.get("doi", ""))
    doi_url = f"https://doi.org/{doi}"
    section = html.escape(section_label(ref)[:48])
    preview = html.escape(chunk_preview(ref, chunks).replace("\n", " "))
    if len(preview) > 400:
        preview = preview[:400] + "…"
    return (
        f'<span class="src-chip">'
        f'<span class="src-idx">{index}</span>'
        f'<a class="src-doi" href="{doi_url}" target="_blank" '
        f'title="{preview}">DOI</a>'
        f'<span class="src-sep">·</span>'
        f'<span class="src-section" title="{preview}">{section}</span>'
        f"</span>"
    )


def sources_bar_html(refs: list[dict], chunks: list[dict]) -> str:
    if not refs:
        return ""
    chips = "".join(
        source_chip_html(ref, chunks, i) for i, ref in enumerate(refs, start=1)
    )
    return f'<div class="src-bar"><span class="src-label">来源</span>{chips}</div>'
