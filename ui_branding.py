"""Visual assets and HTML snippets for the RAG web UI."""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent
LOGO_FULL_FILE = _ASSETS_DIR / "logo.png"
LOGO_ICON_FILE = _ASSETS_DIR / "纯logo.png"
LOGO_FILE = LOGO_FULL_FILE


def _svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


BRAND_LOGO_SVG = _svg_data_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
    '<circle cx="20" cy="20" r="20" fill="#0d9488"/>'
    '<circle cx="20" cy="20" r="13" stroke="#fff" stroke-width="1.5" fill="none" opacity="0.9"/>'
    '<circle cx="20" cy="20" r="3" fill="#fff"/>'
    '<path d="M20 7v6M20 27v6M7 20h6M27 20h6" stroke="#fff" stroke-width="2" stroke-linecap="round"/>'
    '<path d="M11 11l4 4M25 25l4 4M29 11l-4 4M15 25l-4 4" stroke="#a7f3d0" stroke-width="1.8" '
    'stroke-linecap="round"/>'
    "</svg>"
)


def _png_data_uri(path: Path) -> str | None:
    if not path.is_file():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


@lru_cache(maxsize=1)
def brand_logo_full_src() -> str:
    """logo.png — 含「电化学大模型」字样，用于品牌主视觉。"""
    return _png_data_uri(LOGO_FULL_FILE) or BRAND_LOGO_SVG


@lru_cache(maxsize=1)
def brand_logo_icon_src() -> str:
    """纯logo.png — 仅图形，用于页脚/加载/图标等。"""
    return _png_data_uri(LOGO_ICON_FILE) or _png_data_uri(LOGO_FULL_FILE) or BRAND_LOGO_SVG


def has_full_logo() -> bool:
    return LOGO_FULL_FILE.is_file()


def has_icon_logo() -> bool:
    return LOGO_ICON_FILE.is_file()


FEATURE_ICONS = {
    "rag": _svg_data_uri(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">'
        '<path d="M4 6h16v12H4z" stroke="#0d9488" stroke-width="1.6" rx="2"/>'
        '<path d="M8 10h8M8 14h5" stroke="#0284c7" stroke-width="1.6" stroke-linecap="round"/>'
        '<circle cx="17" cy="17" r="4" fill="#e0f2fe" stroke="#0284c7" stroke-width="1.4"/>'
        '<path d="M15.5 17l1 1 2.5-2.5" stroke="#0284c7" stroke-width="1.3" stroke-linecap="round"/>'
        "</svg>"
    ),
    "cite": _svg_data_uri(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">'
        '<path d="M6 4h12a2 2 0 0 1 2 2v14l-4-3-4 3-4-3-4 3V6a2 2 0 0 1 2-2z" '
        'stroke="#0d9488" stroke-width="1.6" fill="#f0fdfa"/>'
        '<path d="M9 9h6M9 12h4" stroke="#0284c7" stroke-width="1.5" stroke-linecap="round"/>'
        "</svg>"
    ),
    "chat": _svg_data_uri(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">'
        '<path d="M4 5h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H9l-5 4V7a2 2 0 0 1 2-2z" '
        'stroke="#0d9488" stroke-width="1.6" fill="#f0fdfa"/>'
        '<path d="M8 10h8M8 13h5" stroke="#0284c7" stroke-width="1.5" stroke-linecap="round"/>'
        "</svg>"
    ),
}


def _fmt_count(value: int | str) -> str:
    return f"{int(value):,}" if value else "—"


def stats_badges_html(
    *,
    oer_papers: int | str,
    oer_chunks: int | str,
    eo_papers: int | str,
    eo_chunks: int | str,
    model: str,
) -> str:
    return (
        '<div class="stats-badges">'
        f'<span class="stat-badge"><i class="dot"></i>OER {_fmt_count(oer_papers)} 篇 · '
        f"{_fmt_count(oer_chunks)} 片段</span>"
        f'<span class="stat-badge"><i class="dot blue"></i>EO {_fmt_count(eo_papers)} 篇 · '
        f"{_fmt_count(eo_chunks)} 片段</span>"
        f'<span class="stat-badge"><i class="dot violet"></i>{model}</span>'
        "</div>"
    )


def feature_cards_html() -> str:
    cards = [
        ("rag", "智能文献检索", "向量语义匹配，精准召回相关章节"),
        ("cite", "可溯源引用", "回答附带 DOI 与原文片段预览"),
        ("chat", "多轮连续对话", "支持追问，保留上下文语境"),
    ]
    parts = ['<div class="feature-grid">']
    for key, title, desc in cards:
        parts.append(
            f'<div class="feature-card">'
            f'<img class="feature-icon" src="{FEATURE_ICONS[key]}" alt=""/>'
            f"<h4>{title}</h4><p>{desc}</p></div>"
        )
    parts.append("</div>")
    return "".join(parts)


def hero_html(
    *,
    oer_papers: int | str,
    oer_chunks: int | str,
    eo_papers: int | str,
    eo_chunks: int | str,
) -> str:
    if has_full_logo():
        logo_block = (
            f'<img class="hero-logo-full" src="{brand_logo_full_src()}" alt="电化学大模型"/>'
        )
        headline = ""
    else:
        logo_block = ""
        headline = "<h2>探索电化学前沿问题</h2>"
    return (
        '<div class="hero-wrap">'
        f"{logo_block}"
        '<div class="hero-copy">'
        '<span class="hero-tag">科研文献助手</span>'
        f"{headline}"
        f"<p>OER 库 <strong>{_fmt_count(oer_papers)}</strong> 篇 / "
        f"<strong>{_fmt_count(oer_chunks)}</strong> 片段 · "
        f"EO 库 <strong>{_fmt_count(eo_papers)}</strong> 篇 / "
        f"<strong>{_fmt_count(eo_chunks)}</strong> 片段</p>"
        "<p>支持 OER、EO 与混合检索，为电催化与电氧化研究提供有据可查的智能解答</p>"
        "</div></div>"
    )


def footer_html(*, index_version: str, ok: bool) -> str:
    status = "服务正常" if ok else "索引待就绪"
    return (
        '<div class="site-footer">'
        '<div class="footer-brand">'
        f'<img class="footer-logo" src="{brand_logo_icon_src()}" alt=""/>'
        "<div>"
        "<strong>电化学大模型</strong>"
        "<span>文献驱动 · 科研辅助</span>"
        "</div></div>"
        f'<p class="footer-meta">索引 {index_version} · {status} · '
        "AI 回答仅供参考，请以原文献为准</p>"
        "</div>"
    )


def loading_banner_html() -> str:
    return (
        '<div class="loading-banner">'
        f'<img class="loading-spin" src="{brand_logo_icon_src()}" alt=""/>'
        '<div><strong>正在检索文献并生成回答</strong>'
        '<span class="typing-dots"><span></span><span></span><span></span></span></div>'
        "</div>"
    )
