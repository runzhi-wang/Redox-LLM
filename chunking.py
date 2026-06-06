"""Structural markdown chunking: sections → paragraphs → merge → sentence split."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from config import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    INDEX_VERSION,
    LOCAL_EMBED_MODEL,
    LOCAL_TOKENIZER_PATH,
)

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_REF_HEADING_RE = re.compile(
    r"^(references?|bibliography|acknowledgements?|supplementary\s+references?|参考文献|致谢)\s*$",
    re.IGNORECASE,
)
_METRICS_RE = re.compile(
    r"(mV|mA\s*cm|mA/cm|overpotential|η\s*@|η10|Tafel|ECSA|mV\s*dec|current\s+density)",
    re.IGNORECASE,
)
_TABLE_CAPTION_RE = re.compile(r"^Table\s+[\d$]", re.IGNORECASE)
_FIG_CAPTION_RE = re.compile(r"^(Fig\.|Figure\s+\d)", re.IGNORECASE)
_HTML_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_HTML_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?…])\s+(?=[A-Z0-9$\\(（【\"'])|(?<=[。！？])\s*"
)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doi: str
    source_file: str
    section: str
    heading_path: str = ""
    h1: str = ""
    h2: str = ""
    h3: str = ""
    h4: str = ""
    index_version: str = INDEX_VERSION
    has_metrics: bool = False
    token_count: int = 0
    content_type: str = "prose"


@dataclass
class _TableBlock:
    caption: str
    html: str
    notes: str = ""


@dataclass
class _FigureBlock:
    text: str


_tokenizer_lock = threading.Lock()
_tokenizer_instance: Optional[object] = None


def _load_tokenizer():
    from transformers import AutoTokenizer

    path = Path(LOCAL_TOKENIZER_PATH)
    if path.is_dir():
        return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    return AutoTokenizer.from_pretrained(LOCAL_EMBED_MODEL, trust_remote_code=True)


def _get_tokenizer():
    global _tokenizer_instance
    if _tokenizer_instance is not None:
        return _tokenizer_instance
    with _tokenizer_lock:
        if _tokenizer_instance is None:
            _tokenizer_instance = _load_tokenizer()
        return _tokenizer_instance


def warmup_tokenizer() -> None:
    """Load tokenizer once in the main thread before parallel indexing."""
    _get_tokenizer()


def doi_from_filename(path: Path) -> str:
    return path.stem.replace("%", "/")


def preprocess_text(text: str) -> str:
    text = _IMAGE_RE.sub("[Figure]", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_before_references(text: str) -> str:
    matches = list(_HEADING_RE.finditer(text))
    for match in matches:
        title = match.group(2).strip()
        if _REF_HEADING_RE.match(title):
            return text[: match.start()].rstrip()
    return text


def _heading_level(marker: str) -> int:
    return len(marker)


def _path_from_stack(stack: dict[int, str]) -> str:
    return " > ".join(stack[i] for i in sorted(stack) if stack.get(i))


def _stack_to_fields(stack: dict[int, str]) -> dict[str, str]:
    return {
        "h1": stack.get(1, ""),
        "h2": stack.get(2, ""),
        "h3": stack.get(3, ""),
        "h4": stack.get(4, ""),
    }


def _encode(text: str) -> list[int]:
    return _get_tokenizer().encode(text, add_special_tokens=False)


def _decode(tokens: list[int]) -> str:
    return _get_tokenizer().decode(tokens, skip_special_tokens=True).strip()


def _token_len(text: str) -> int:
    if not text.strip():
        return 0
    return len(_encode(text))


def _join_parts(parts: list[str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def _tail_overlap(text: str) -> str:
    if CHUNK_OVERLAP_TOKENS <= 0:
        return ""
    tokens = _encode(text)
    if len(tokens) <= CHUNK_OVERLAP_TOKENS:
        return text.strip()
    return _decode(tokens[-CHUNK_OVERLAP_TOKENS:])


def _has_metrics(text: str) -> bool:
    return bool(_METRICS_RE.search(text))


def _strip_html(cell: str) -> str:
    cell = re.sub(r"<[^>]+>", "", cell)
    return re.sub(r"\s+", " ", cell).strip()


def _parse_html_table(html: str) -> tuple[list[str], list[str]]:
    rows = _HTML_ROW_RE.findall(html)
    if not rows:
        return [], [html]

    parsed_rows: list[list[str]] = []
    for row_html in rows:
        cells = [_strip_html(c) for c in _HTML_CELL_RE.findall(row_html)]
        if cells:
            parsed_rows.append(cells)

    if not parsed_rows:
        return [], [html]

    headers = parsed_rows[0]
    data_lines: list[str] = []
    for cells in parsed_rows[1:]:
        data_lines.append(" | ".join(cells))
    return headers, data_lines


def _table_context_header(doi: str, section: str, heading_path: str) -> str:
    return f"DOI: {doi}\nSection: {section}\nHeading: {heading_path}\n"


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def _token_chunks_with_overlap(text: str) -> list[str]:
    """Fallback when a single sentence exceeds MAX_CHUNK_TOKENS."""
    tokens = _encode(text)
    if len(tokens) <= CHUNK_MAX_TOKENS:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_MAX_TOKENS, len(tokens))
        piece = _decode(tokens[start:end])
        if piece.strip():
            chunks.append(piece.strip())
        if end >= len(tokens):
            break
        start = max(start + 1, end - CHUNK_OVERLAP_TOKENS)
    return chunks


def _split_with_sentence_overlap(text: str) -> list[str]:
    """Split long prose at sentence boundaries with token-level tail overlap."""
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    overlap_prefix = ""
    idx = 0

    while idx < len(sentences):
        parts: list[str] = []
        if overlap_prefix:
            parts.append(overlap_prefix)

        while idx < len(sentences):
            sentence = sentences[idx]
            st = _token_len(sentence)

            if st > CHUNK_MAX_TOKENS and not parts:
                chunks.extend(_token_chunks_with_overlap(sentence))
                overlap_prefix = _tail_overlap(chunks[-1]) if chunks else ""
                idx += 1
                break

            trial = _join_parts(parts + [sentence])
            tt = _token_len(trial)

            if tt > CHUNK_MAX_TOKENS and parts:
                break

            parts.append(sentence)
            idx += 1

            if tt >= CHUNK_MAX_TOKENS:
                break

        body = _join_parts(parts)
        if body:
            chunks.append(body)
            overlap_prefix = _tail_overlap(body)
        else:
            overlap_prefix = ""

    return _merge_tiny_tail(chunks)


def _merge_tiny_tail(chunks: list[str]) -> list[str]:
    if len(chunks) < 2:
        return chunks
    if _token_len(chunks[-1]) >= CHUNK_MIN_TOKENS:
        return chunks
    merged = chunks[-2] + "\n\n" + chunks[-1]
    if _token_len(merged) <= CHUNK_MAX_TOKENS:
        return chunks[:-2] + [merged]
    return chunks


def _emit_merged_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if _token_len(text) > CHUNK_MAX_TOKENS:
        return _split_with_sentence_overlap(text)
    return [text]


def _chunk_prose_paragraphs(paragraphs: list[str]) -> list[str]:
    """
    Merge complete paragraphs within a section up to MAX tokens.
    Flush only when the next paragraph would exceed MAX.
    """
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def emit_buffer() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        merged = "\n\n".join(buffer)
        chunks.extend(_emit_merged_text(merged))
        buffer = []
        buffer_tokens = 0

    for para in paragraphs:
        pt = _token_len(para)

        if pt > CHUNK_MAX_TOKENS:
            emit_buffer()
            chunks.extend(_emit_merged_text(para))
            continue

        if buffer_tokens + pt > CHUNK_MAX_TOKENS:
            emit_buffer()

        buffer.append(para)
        buffer_tokens += pt

    emit_buffer()
    return _merge_tiny_tail(chunks)


def _chunk_table_block(
    block: _TableBlock,
    *,
    doi: str,
    section: str,
    heading_path: str,
) -> list[str]:
    headers, data_lines = _parse_html_table(block.html)
    ctx = _table_context_header(doi, section, heading_path)
    caption = block.caption.strip()
    notes = block.notes.strip()
    notes_block = f"\n\nTable notes:\n{notes}" if notes else ""

    if not data_lines:
        body = block.html if block.html else caption
        return [f"{ctx}\n{caption}\n\n{body}{notes_block}".strip()]

    header_line = " | ".join(headers) if headers else "(column headers in table)"
    prefix = f"{ctx}\n{caption}\n\nColumn headers:\n{header_line}\n\nTable rows:\n"

    chunks: list[str] = []
    current_rows: list[str] = []

    def emit_rows() -> None:
        if not current_rows:
            return
        text = prefix + "\n".join(current_rows) + notes_block
        chunks.append(text.strip())

    for row in data_lines:
        trial_rows = current_rows + [row]
        trial = prefix + "\n".join(trial_rows) + notes_block
        if _token_len(trial) > CHUNK_MAX_TOKENS and current_rows:
            emit_rows()
            current_rows = [row]
        else:
            current_rows.append(row)

    emit_rows()
    if not chunks:
        chunks.append(f"{prefix}{notes_block}".strip())
    return chunks


def _is_table_note_line(line: str, caption: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if lower.startswith(("note:", "notes:", "a ", "the values", "all ")):
        return True
    table_ref = re.search(r"table\s+[\d$]", caption, re.IGNORECASE)
    if table_ref and table_ref.group(0).lower() in lower:
        return True
    return False


def _extract_html_table(lines: list[str], start: int) -> tuple[str, int]:
    if start >= len(lines):
        return "", start
    line = lines[start].strip()
    if not line.lower().startswith("<table"):
        return "", start

    if line.endswith("</table>"):
        return line, start + 1

    parts = [line]
    idx = start + 1
    while idx < len(lines):
        parts.append(lines[idx])
        if "</table>" in lines[idx].lower():
            idx += 1
            break
        idx += 1
    return "\n".join(parts), idx


def _collect_table_notes(lines: list[str], start: int, caption: str) -> tuple[str, int]:
    notes: list[str] = []
    idx = start
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            if notes:
                break
            continue
        if (
            _TABLE_CAPTION_RE.match(line)
            or _FIG_CAPTION_RE.match(line)
            or line.startswith("<table")
            or _HEADING_RE.match(line)
        ):
            break
        if _is_table_note_line(line, caption) or (notes and _token_len(line) < 120):
            notes.append(line)
            idx += 1
            continue
        break
    return "\n".join(notes), idx


def _parse_section_blocks(body: str) -> list[Literal["prose", "table", "figure"] | object]:
    body = preprocess_text(body)
    if not body:
        return []

    lines = body.split("\n")
    blocks: list[object] = []
    prose_buffer: list[str] = []
    idx = 0

    def flush_prose() -> None:
        nonlocal prose_buffer
        if prose_buffer:
            blocks.append(list(prose_buffer))
            prose_buffer = []

    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        if _TABLE_CAPTION_RE.match(line):
            flush_prose()
            caption = line
            idx += 1
            html, idx = _extract_html_table(lines, idx)
            notes, idx = _collect_table_notes(lines, idx, caption)
            blocks.append(_TableBlock(caption=caption, html=html, notes=notes))
            continue

        if line.lower().startswith("<table"):
            flush_prose()
            html, idx = _extract_html_table(lines, idx)
            blocks.append(_TableBlock(caption="(table)", html=html))
            continue

        if line == "[Figure]" or _FIG_CAPTION_RE.match(line):
            flush_prose()
            fig_parts = [line]
            idx += 1
            if idx < len(lines) and _FIG_CAPTION_RE.match(lines[idx].strip()):
                fig_parts.append(lines[idx].strip())
                idx += 1
            elif line == "[Figure]" and idx < len(lines) and not lines[idx].strip().startswith("<"):
                next_line = lines[idx].strip()
                if next_line and not _TABLE_CAPTION_RE.match(next_line):
                    fig_parts.append(next_line)
                    idx += 1
            blocks.append(_FigureBlock(text="\n".join(fig_parts)))
            continue

        para_lines = [line]
        idx += 1
        while idx < len(lines):
            nxt = lines[idx].strip()
            if not nxt:
                idx += 1
                break
            if (
                _TABLE_CAPTION_RE.match(nxt)
                or nxt.lower().startswith("<table")
                or nxt == "[Figure]"
                or _FIG_CAPTION_RE.match(nxt)
            ):
                break
            para_lines.append(nxt)
            idx += 1
        prose_buffer.append("\n".join(para_lines))

    flush_prose()
    return blocks


def _split_section_body(
    body: str,
    *,
    doi: str,
    section: str,
    heading_path: str,
) -> list[tuple[str, str]]:
    pieces: list[tuple[str, str]] = []
    blocks = _parse_section_blocks(body)

    for block in blocks:
        if isinstance(block, list):
            for prose in _chunk_prose_paragraphs(block):
                if prose.strip():
                    pieces.append((prose, "prose"))
        elif isinstance(block, _TableBlock):
            for table_chunk in _chunk_table_block(
                block, doi=doi, section=section, heading_path=heading_path
            ):
                if table_chunk.strip():
                    pieces.append((table_chunk, "table"))
        elif isinstance(block, _FigureBlock):
            text = block.text.strip()
            if not text:
                continue
            if _token_len(text) > CHUNK_MAX_TOKENS:
                for piece in _split_with_sentence_overlap(text):
                    pieces.append((piece, "figure"))
            else:
                header = _table_context_header(doi, section, heading_path)
                pieces.append((f"{header}{text}", "figure"))

    return pieces


def chunk_markdown(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    text = truncate_before_references(raw)
    text = preprocess_text(text)
    doi = doi_from_filename(path)
    source_file = path.name

    matches = list(_HEADING_RE.finditer(text))
    sections: list[tuple[dict[int, str], str, str]] = []

    if not matches:
        sections.append(({}, "document", text))
    else:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(({}, "abstract", preamble))

        stack: dict[int, str] = {}
        for i, match in enumerate(matches):
            level = _heading_level(match.group(1))
            title = match.group(2).strip()
            for lv in list(stack):
                if lv >= level:
                    del stack[lv]
            stack[level] = title

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((dict(stack), title, body))

    chunks: list[Chunk] = []
    chunk_idx = 0
    for stack, section_title, body in sections:
        fields = _stack_to_fields(stack)
        heading_path = _path_from_stack(stack) or section_title
        for piece, content_type in _split_section_body(
            body,
            doi=doi,
            section=section_title,
            heading_path=heading_path,
        ):
            if not piece.strip():
                continue
            chunk_idx += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doi}#{INDEX_VERSION}#{chunk_idx}",
                    text=piece,
                    doi=doi,
                    source_file=source_file,
                    section=section_title,
                    heading_path=heading_path,
                    h1=fields["h1"],
                    h2=fields["h2"],
                    h3=fields["h3"],
                    h4=fields["h4"],
                    has_metrics=_has_metrics(piece),
                    token_count=_token_len(piece),
                    content_type=content_type,
                )
            )
    return chunks
