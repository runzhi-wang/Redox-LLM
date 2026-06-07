"""电化学大模型 — 用户优先的 RAG 对话界面."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st

from ask import query_rag
from chat_log import append_record, export_xlsx, list_records, update_feedback
from chroma_utils import create_chroma_client, vector_index_healthy
from chromadb.errors import NotFoundError
from config import (
    CHAT_LOG_EXPORT_PATH,
    CHAT_MODEL,
    CHAT_MODEL_OPTIONS,
    CHROMA_DIR,
    INDEX_VERSION,
    TOP_K,
    get_client,
)
from corpora import CORPORA, RAG_MODE_LABELS, count_indexed_papers, mode_ready
from citations import chunk_preview, section_label
from ui_branding import (
    LOGO_FULL_FILE,
    LOGO_ICON_FILE,
    brand_logo_full_src,
    brand_logo_icon_src,
    feature_cards_html,
    footer_html,
    has_full_logo,
    hero_html,
    loading_banner_html,
    stats_badges_html,
)

EXAMPLES = {
    "oer": [
        "中性 pH 下如何同时提升 OER 稳定性并降低过电位？",
        "NiFe 催化剂在中性介质中的活性与机理？",
        "哪些设计思路可迁移到电芬顿电极？",
    ],
    "eo": [
        "电氧化降解有机污染物时活性氯物种起什么作用？",
        "BDD 电极在 EO 处理中的优势与局限？",
        "哪些因素决定 EO 体系的电流效率？",
    ],
    "mixed": [
        "OER 催化剂设计思路能否用于 EO 阳极？",
        "电芬顿与阳极氧化联用的文献进展？",
        "中性介质下如何选择 OER 或 EO 相关策略？",
    ],
}
RAG_MODE_OPTIONS = list(RAG_MODE_LABELS.keys())

MAX_HISTORY_TURNS = 8

RATING_LABELS = {"good": "满意 👍", "fair": "一般", "bad": "需改进 👎"}

USER_AVATAR = "data:image/svg+xml;base64," + base64.b64encode(
    (
        '<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">'
        '<circle cx="18" cy="18" r="18" fill="#e8f4fc"/>'
        '<circle cx="9.5" cy="9" r="6.5" fill="#1a1a2e" stroke="#1a1a2e" stroke-width="1.2"/>'
        '<circle cx="26.5" cy="9" r="6.5" fill="#1a1a2e" stroke="#1a1a2e" stroke-width="1.2"/>'
        '<circle cx="10.5" cy="10" r="2.2" fill="#ffc9d9"/>'
        '<circle cx="25.5" cy="10" r="2.2" fill="#ffc9d9"/>'
        '<ellipse cx="18" cy="20.5" rx="13.5" ry="12.5" fill="#fff" '
        'stroke="#1a1a2e" stroke-width="1.4"/>'
        '<ellipse cx="11.5" cy="18.5" rx="5" ry="5.5" fill="#1a1a2e"/>'
        '<ellipse cx="24.5" cy="18.5" rx="5" ry="5.5" fill="#1a1a2e"/>'
        '<circle cx="12.5" cy="17" r="2.6" fill="#fff"/>'
        '<circle cx="25.5" cy="17" r="2.6" fill="#fff"/>'
        '<circle cx="13.2" cy="17.2" r="1.5" fill="#1a1a2e"/>'
        '<circle cx="26.2" cy="17.2" r="1.5" fill="#1a1a2e"/>'
        '<circle cx="14" cy="16.2" r="0.7" fill="#fff"/>'
        '<circle cx="27" cy="16.2" r="0.7" fill="#fff"/>'
        '<ellipse cx="18" cy="22.5" rx="2.2" ry="1.6" fill="#1a1a2e"/>'
        '<path d="M13.5 25.5 Q18 29.5 22.5 25.5" fill="none" stroke="#1a1a2e" '
        'stroke-width="1.3" stroke-linecap="round"/>'
        '<circle cx="8.5" cy="24" r="2.5" fill="#ffb3c6" opacity="0.75"/>'
        '<circle cx="27.5" cy="24" r="2.5" fill="#ffb3c6" opacity="0.75"/>'
        '</svg>'
    ).encode()
).decode()

BOT_AVATAR = "data:image/svg+xml;base64," + base64.b64encode(
    (
        '<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">'
        '<circle cx="18" cy="18" r="18" fill="#10a37f"/>'
        '<path fill="#fff" d="M18 10c-2.4 0-4.4 1.4-5.3 3.4-1.8-.5-3.6.8-4 2.7-.4 1.9 1 3.6 2.8 4.1-.3 2.2 1.4 4.2 3.6 4.4 '
        '2.2.2 4-1.3 4.5-3.3 1.7.7 3.7-.2 4.4-2 .7-1.8-.2-3.7-1.9-4.6.4-2.3-.4-4.7-2.1-5.8C21.2 10.2 19.6 10 18 10zm0 2c1 0 1.9.4 2.5 1.1-1 .3-1.8.8-2.4 1.4-.6-.9-1.5-1.5-2.5-1.5-1 0-1.8.5-2.4 1.3.6.4 1.1.9 1.5 1.5.7-.8 '
        '1.6-1.3 2.6-1.5.3-.7.5-1.4.7-2.3zm4.8 2.2c.7.6 1.1 1.5 1 2.5-.8-.2-1.5-.6-2.1-1.1.3-.5.7-.9 1.1-1.4zm-9.6 0c.4.5.8.9 1.1 1.4-.6.5-1.3.9-2.1 1.1-.1-1 .3-1.9 1-2.5zm4.8 1.5c1.2 0 2.2.8 2.5 2-.7.4-1.5.6-2.4.6-.9 0-1.7-.2-2.4-.6.3-1.2 1.3-2 2.5-2zm0 4.8c-1.4 0-2.6-.8-3.1-2 .9.5 1.9.7 3 .7 1.1 0 2.1-.2 3-.7-.5 1.2-1.7 2-3.1 2z"/>'
        '</svg>'
    ).encode()
).decode()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stStatusWidget"] { display: none !important; }
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
        /* 避免点击后整页短暂变灰 */
        .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        section[data-testid="stMain"], .main {
            opacity: 1 !important;
            transition: none !important;
        }
        .stApp[data-testscript-state="running"] {
            opacity: 1 !important;
        }
        [data-testid="stVerticalBlock"] {
            opacity: 1 !important;
        }
        .stApp {
            background:
                radial-gradient(ellipse 80% 55% at 0% 0%, rgba(13, 148, 136, 0.07), transparent 55%),
                radial-gradient(ellipse 70% 50% at 100% 8%, rgba(59, 130, 246, 0.06), transparent 50%),
                linear-gradient(180deg, #fafcff 0%, #ffffff 42%, #f4f8fc 100%);
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .main {
            background: transparent !important;
        }
        /* PC 桌面端：中间内容 55vw，左右各约 22.5% 留白 */
        .main .block-container {
            width: 55vw !important;
            max-width: 55vw !important;
            min-width: 55vw !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding: 4.75rem 0 11rem;
        }
        [data-testid="stMainBlockContainer"] {
            width: 55vw !important;
            max-width: 55vw !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.82) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(226, 235, 245, 0.95) !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 32px rgba(15, 23, 42, 0.05) !important;
        }
        .site-brand {
            position: fixed;
            top: 0.6rem;
            left: calc((100vw - 55vw) / 2);
            z-index: 1000;
            text-align: left;
            pointer-events: none;
        }
        .site-brand .brand-title,
        .site-brand .brand-status {
            pointer-events: auto;
        }
        .toolbar-spacer {
            height: 3.25rem;
            margin-bottom: 0.35rem;
            border-bottom: 1px solid rgba(226, 235, 245, 0.9);
        }
        .main .block-container [data-testid="stHorizontalBlock"]:has(.hdr-btn) {
            position: fixed !important;
            top: 0.7rem !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            width: 55vw !important;
            max-width: 55vw !important;
            z-index: 999 !important;
            margin: 0 !important;
            padding: 0 0.15rem !important;
            justify-content: flex-end !important;
            overflow: visible !important;
            background: transparent !important;
            pointer-events: none !important;
        }
        .main .block-container [data-testid="stHorizontalBlock"]:has(.hdr-btn) > [data-testid="column"] {
            pointer-events: auto !important;
        }
        [data-testid="stBottom"] {
            bottom: 18px !important;
            background: transparent !important;
        }
        [data-testid="stBottomBlockContainer"] {
            width: 55vw !important;
            max-width: 55vw !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding: 0.7rem 1.25rem 0.85rem;
            background: rgba(255, 255, 255, 0.94) !important;
            backdrop-filter: blur(14px);
            border: 1px solid rgba(210, 222, 235, 0.95);
            border-bottom: none;
            border-radius: 16px 16px 0 0;
            box-shadow: 0 -10px 36px rgba(15, 23, 42, 0.08);
        }
        [data-testid="stBottom"] > div {
            width: 55vw !important;
            max-width: 55vw !important;
            margin: 0 auto;
            background: transparent !important;
        }
        .top-bar {
            display: flex; align-items: center; justify-content: space-between;
            padding: 0.35rem 0 0.85rem;
            border-bottom: 1px solid #e8ecf1;
            margin-bottom: 0.75rem;
        }
        .brand-title {
            font-family: "STKaiti", "KaiTi", "FangSong", "STSong", "Noto Serif SC", serif;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            margin: 0;
            line-height: 1.15;
            white-space: nowrap;
            background: linear-gradient(120deg, #0c4a6e 0%, #0369a1 38%, #0d9488 72%, #0284c7 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 1px 1px rgba(12, 74, 110, 0.22));
        }
        .brand-status {
            font-size: 0.78rem; color: #5b6b7c; margin: 0.2rem 0 0;
            letter-spacing: 0.02em;
        }
        .status-ok { color: #059669; }
        .status-bad { color: #dc2626; }
        .site-brand-inner {
            display: flex; align-items: center; gap: 0.55rem;
        }
        .site-brand-inner.site-brand-full {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.2rem;
        }
        .brand-logo-icon {
            height: 2.5rem;
            width: auto;
            max-width: 3.5rem;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
        }
        .brand-logo-full {
            height: 3.1rem;
            width: auto;
            max-width: 14rem;
            object-fit: contain;
            display: block;
        }
        .hero-logo-full {
            height: 4.5rem;
            width: auto;
            max-width: 20rem;
            object-fit: contain;
            margin-bottom: 0.75rem;
        }
        .hero-wrap {
            display: flex; flex-direction: column; align-items: center;
            padding: 0.75rem 0.5rem 0.5rem; text-align: center;
        }
        .hero-copy h2 {
            font-size: 1.45rem; font-weight: 650; color: #0f172a;
            margin: 0.35rem 0 0.5rem; letter-spacing: 0.02em;
        }
        .hero-copy p {
            color: #5b6b7c; font-size: 0.92rem; margin: 0;
            line-height: 1.6; max-width: 100%;
        }
        .hero-tag {
            display: inline-block; font-size: 0.72rem; font-weight: 600;
            letter-spacing: 0.08em; text-transform: uppercase;
            color: #0f766e; background: #ecfdf5;
            border: 1px solid #99f6e4; border-radius: 999px;
            padding: 0.2rem 0.65rem;
        }
        .stats-badges {
            display: flex; flex-wrap: wrap; justify-content: center;
            gap: 0.45rem; margin: 1rem 0 0.85rem;
        }
        .stat-badge {
            display: inline-flex; align-items: center; gap: 0.35rem;
            font-size: 0.78rem; color: #475569;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #dce6f0; border-radius: 999px;
            padding: 0.28rem 0.7rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }
        .stat-badge .dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: #10b981; display: inline-block;
        }
        .stat-badge .dot.blue { background: #3b82f6; }
        .stat-badge .dot.violet { background: #8b5cf6; }
        .feature-grid {
            display: grid; grid-template-columns: repeat(3, 1fr);
            gap: 0.65rem; margin: 0.5rem 0 1rem;
        }
        .feature-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #e2ebf4; border-radius: 12px;
            padding: 0.85rem 0.75rem; text-align: center;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .feature-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
        }
        .feature-icon { width: 2rem; height: 2rem; margin-bottom: 0.35rem; }
        .feature-card h4 {
            font-size: 0.86rem; font-weight: 600; color: #1e293b;
            margin: 0 0 0.25rem;
        }
        .feature-card p {
            font-size: 0.74rem; color: #7b8a9a; margin: 0; line-height: 1.45;
        }
        .example-section-title {
            font-size: 0.82rem; font-weight: 600; color: #64748b;
            margin: 0.25rem 0 0.45rem; letter-spacing: 0.04em;
        }
        div.example-list [data-testid="stButton"] button,
        div.hdr-btn [data-testid="stButton"] button {
            text-align: left !important;
            background: rgba(255, 255, 255, 0.95) !important;
            border: 1px solid #d7e3ef !important;
            border-radius: 10px !important;
            padding: 0.65rem 0.85rem !important;
            color: #334155 !important;
            font-size: 0.88rem !important;
            line-height: 1.45 !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
            transition: border-color 0.15s ease, color 0.15s ease !important;
        }
        div.hdr-btn [data-testid="stButton"] button {
            text-align: center !important;
            padding: 0.35rem 0.55rem !important;
            font-size: 0.84rem !important;
        }
        div.example-list [data-testid="stButton"] button:hover,
        div.hdr-btn [data-testid="stButton"] button:hover {
            border-color: #0d9488 !important;
            color: #0f766e !important;
            background: rgba(255, 255, 255, 0.95) !important;
        }
        div.example-list [data-testid="stButton"] button:active,
        div.hdr-btn [data-testid="stButton"] button:active,
        div.example-list [data-testid="stButton"] button:focus,
        div.hdr-btn [data-testid="stButton"] button:focus {
            background: rgba(255, 255, 255, 0.95) !important;
            color: #0f766e !important;
            border-color: #0d9488 !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
        }
        .loading-banner {
            display: flex; align-items: center; gap: 0.75rem;
            padding: 0.65rem 0.85rem; margin: 0.25rem 0;
            background: linear-gradient(90deg, #f0fdfa, #eff6ff);
            border: 1px solid #cce8e4; border-radius: 12px;
        }
        .loading-banner strong {
            display: block; font-size: 0.86rem; color: #0f766e;
        }
        .loading-spin {
            width: 2rem; height: 2rem;
            animation: spin 2.5s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .site-footer {
            margin-top: 1.5rem;
            margin-bottom: 6rem;
            padding: 1rem 0 0.5rem;
            border-top: 1px solid rgba(226, 235, 245, 0.95);
        }
        .footer-brand {
            display: flex; align-items: center; gap: 0.55rem;
            margin-bottom: 0.35rem;
        }
        .footer-logo {
            height: 1.75rem;
            width: auto;
            max-width: 3rem;
            object-fit: contain;
            border-radius: 4px;
        }
        .footer-brand strong {
            display: block; font-size: 0.88rem; color: #1e293b;
        }
        .footer-brand span {
            font-size: 0.74rem; color: #94a3b8;
        }
        .footer-meta {
            font-size: 0.72rem; color: #94a3b8; margin: 0;
            line-height: 1.5;
        }
        .page-banner {
            display: flex; align-items: center; gap: 0.6rem;
            padding: 0.65rem 0.85rem; margin-bottom: 0.85rem;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #e2ebf4; border-radius: 12px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        }
        .page-banner img { width: 1.75rem; height: 1.75rem; }
        .page-banner strong { font-size: 0.95rem; color: #1e293b; }
        .page-banner span { font-size: 0.78rem; color: #7b8a9a; }
        [data-testid="stChatMessage"] {
            padding: 0.35rem 0.5rem;
            margin: 0.15rem 0;
            border-radius: 12px;
        }
        [data-testid="stChatMessage"] p {
            font-size: 0.96rem; line-height: 1.65; color: #1e293b;
        }
        .assistant-meta { font-size: 0.75rem; color: #7b8a9a; margin-top: 0.35rem; }
        [data-testid="stChatInput"] {
            width: 100%;
        }
        [data-testid="stChatInput"] > div {
            display: flex !important;
            align-items: center !important;
            width: 100% !important;
            background: #ffffff !important;
            border: 1px solid #c5d3e3 !important;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
            padding: 6px 8px 6px 6px;
        }
        [data-testid="stChatInput"] textarea {
            flex: 1 1 auto !important;
            width: 100% !important;
            min-width: 0 !important;
            min-height: 52px !important;
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
            padding: 13px 10px !important;
            border: none !important;
            background: transparent !important;
            color: #1e293b !important;
            resize: none !important;
            box-sizing: border-box !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            font-size: 0.92rem !important;
            color: #64748b !important;
            line-height: 1.5 !important;
        }
        [data-testid="stChatInputSubmitButton"],
        [data-testid="stChatInputSubmitButton"] button {
            flex: 0 0 auto !important;
            align-self: center !important;
            min-width: 2.6rem !important;
            width: 2.6rem !important;
            height: 2.6rem !important;
            padding: 0 !important;
            margin: 0 !important;
            border-radius: 10px !important;
            background: linear-gradient(135deg, #0d9488, #0284c7) !important;
            border: none !important;
            outline: none !important;
            box-shadow: 0 2px 10px rgba(13, 148, 136, 0.38) !important;
            color: #ffffff !important;
        }
        [data-testid="stChatInputSubmitButton"]:hover:not(:disabled),
        [data-testid="stChatInputSubmitButton"] button:hover:not(:disabled) {
            background: linear-gradient(135deg, #0f766e, #0369a1) !important;
            color: #ffffff !important;
            border: none !important;
            outline: none !important;
            box-shadow: 0 2px 10px rgba(13, 148, 136, 0.42) !important;
        }
        [data-testid="stChatInputSubmitButton"]:focus,
        [data-testid="stChatInputSubmitButton"]:focus-visible,
        [data-testid="stChatInputSubmitButton"] button:focus,
        [data-testid="stChatInputSubmitButton"] button:focus-visible {
            border: none !important;
            outline: none !important;
            box-shadow: 0 2px 10px rgba(13, 148, 136, 0.38) !important;
        }
        [data-testid="stChatInputSubmitButton"]:disabled,
        [data-testid="stChatInputSubmitButton"] button:disabled {
            background: #cbd5e1 !important;
            color: #f8fafc !important;
            border: none !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInputSubmitButton"] svg,
        [data-testid="stChatInputSubmitButton"] svg path {
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }
        .typing-dots span {
            display: inline-block; width: 6px; height: 6px; margin: 0 2px;
            background: #94a3b8; border-radius: 50%;
            animation: blink 1.2s infinite;
        }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes blink {
            0%,80%,100% { opacity: 0.25; }
            40% { opacity: 1; }
        }
        [data-testid="stHorizontalBlock"]:has(.hdr-btn) {
            flex-wrap: nowrap !important;
            align-items: center !important;
            gap: 0.35rem !important;
            overflow: visible !important;
        }
        [data-testid="stHorizontalBlock"]:has(.hdr-btn) > [data-testid="column"] {
            flex: 0 0 auto !important;
            min-width: 0;
        }
        [data-testid="stHorizontalBlock"]:has(.hdr-btn) .hdr-btn button,
        [data-testid="stHorizontalBlock"]:has(.hdr-btn) [data-testid="stPopover"] button {
            font-size: 0.84rem !important;
            border-radius: 8px !important;
            padding: 0.35rem 0.55rem !important;
            border: 1px solid #d5e0ec !important;
            background: rgba(255, 255, 255, 0.92) !important;
            color: #334155 !important;
            white-space: nowrap !important;
            word-break: keep-all !important;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04) !important;
        }
        [data-testid="stHorizontalBlock"]:has(.hdr-btn) .hdr-btn button:hover,
        [data-testid="stHorizontalBlock"]:has(.hdr-btn) [data-testid="stPopover"] button:hover {
            border-color: #0d9488 !important;
            color: #0f766e !important;
        }
        [data-testid="stHorizontalBlock"]:has(.hdr-btn) > [data-testid="column"]:last-child {
            min-width: 3.6rem;
        }
        .feedback-panel {
            margin: 0.7rem 0 0.35rem;
            padding: 0.8rem 0.9rem;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #dce6f0;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        }
        .feedback-panel-title {
            font-size: 0.9rem; font-weight: 600; color: #334155; margin: 0 0 0.15rem;
        }
        .feedback-panel-hint {
            font-size: 0.78rem; color: #94a3b8; margin: 0 0 0.5rem;
        }
        .feedback-done {
            font-size: 0.82rem; color: #059669; margin: 0 0 0.45rem;
            font-weight: 500;
        }
        div.feedback-panel [data-testid="stHorizontalBlock"] button {
            font-size: 0.86rem !important;
            border-radius: 8px !important;
            padding: 0.42rem 0.5rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _read_index_report() -> Optional[dict]:
    p = Path(__file__).resolve().parent / "output" / "index_report.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _corpus_stats(client, key: str) -> dict:
    corpus = CORPORA[key]
    papers = count_indexed_papers(corpus)
    chunks = 0
    ok = False
    try:
        col = client.get_collection(corpus.collection_name)
        chunks = int(col.count())
        ok = chunks > 0 and vector_index_healthy(col)
    except NotFoundError:
        pass
    except Exception:  # noqa: BLE001
        pass
    if not papers:
        report = _read_index_report()
        if report and key == "oer":
            papers = int(report.get("files_done") or report.get("files_total") or 0)
    return {"papers": papers, "chunks": chunks, "ok": ok}


@st.cache_data(ttl=30, show_spinner=False)
def _index_status() -> dict:
    empty = {k: {"papers": count_indexed_papers(CORPORA[k]), "chunks": 0, "ok": False} for k in CORPORA}
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        return {**empty, "ok": False, "papers": 0, "chunks": 0}
    try:
        client = create_chroma_client()
        try:
            stats = {key: _corpus_stats(client, key) for key in CORPORA}
        finally:
            client.close()
    except Exception:  # noqa: BLE001
        stats = empty
    oer = stats.get("oer", empty["oer"])
    eo = stats.get("eo", empty["eo"])
    return {
        "oer": oer,
        "eo": eo,
        "papers": int(oer.get("papers", 0)) + int(eo.get("papers", 0)),
        "chunks": int(oer.get("chunks", 0)) + int(eo.get("chunks", 0)),
        "ok": bool(oer.get("ok") or eo.get("ok")),
    }


def _fmt_time(iso: str) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        else:
            dt = dt.replace(tzinfo=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso[:16]


def _llm_history(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] in ("user", "assistant") and m.get("content"):
            out.append({"role": m["role"], "content": m["content"]})
    return out[-MAX_HISTORY_TURNS:]


def _render_sources(refs: list[dict], chunks: list[dict]) -> None:
    if not refs:
        return
    st.caption("参考文献（点击章节可预览片段）")
    n = min(len(refs), 4)
    cols = st.columns(n)
    for i, ref in enumerate(refs[:n]):
        with cols[i]:
            title = section_label(ref)[:22] + ("…" if len(section_label(ref)) > 22 else "")
            preview = chunk_preview(ref, chunks, 500)
            with st.popover(title):
                st.link_button("DOI 原文", f"https://doi.org/{ref['doi']}", use_container_width=True)
                if preview:
                    st.text(preview)
                else:
                    st.caption("暂无预览")
    if len(refs) > n:
        with st.popover(f"更多 +{len(refs) - n}"):
            for j, ref in enumerate(refs[n:], start=n + 1):
                st.markdown(f"**{j}.** [{ref['doi']}](https://doi.org/{ref['doi']})")
                st.caption(section_label(ref))
                prev = chunk_preview(ref, chunks, 300)
                if prev:
                    st.text(prev[:300])


def _apply_feedback(msg: dict, *, rating: str, comment: Optional[str] = None) -> bool:
    meta = msg.get("meta") or {}
    record_id = meta.get("record_id")
    if not record_id:
        return False
    kwargs: dict = {"rating": rating}
    if comment is not None:
        kwargs["comment"] = comment
    if not update_feedback(record_id, **kwargs):
        return False
    meta["rating"] = rating
    if comment is not None:
        meta["comment"] = comment.strip()
    msg["meta"] = meta
    return True


def _render_feedback(msg: dict, idx: int) -> None:
    meta = msg.get("meta") or {}
    record_id = meta.get("record_id")
    if not record_id:
        return

    rating = meta.get("rating") or ""
    comment = meta.get("comment") or ""

    st.markdown('<div class="feedback-panel">', unsafe_allow_html=True)
    st.markdown(
        '<p class="feedback-panel-title">评价此回答</p>'
        '<p class="feedback-panel-hint">在此直接反馈，无需打开提问记录</p>',
        unsafe_allow_html=True,
    )
    if rating:
        st.markdown(
            f'<p class="feedback-done">已评价：{RATING_LABELS.get(rating, rating)}</p>',
            unsafe_allow_html=True,
        )

    note_key = f"fb_note_{idx}"

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("👍 满意", key=f"fb_good_{idx}", use_container_width=True):
            note_val = st.session_state.get(note_key, comment)
            if _apply_feedback(msg, rating="good", comment=note_val):
                st.toast("感谢反馈！")
                st.rerun()
    with c2:
        if st.button("😐 一般", key=f"fb_fair_{idx}", use_container_width=True):
            note_val = st.session_state.get(note_key, comment)
            if _apply_feedback(msg, rating="fair", comment=note_val):
                st.toast("感谢反馈！")
                st.rerun()
    with c3:
        if st.button("👎 需改进", key=f"fb_bad_{idx}", use_container_width=True):
            note_val = st.session_state.get(note_key, comment)
            if _apply_feedback(msg, rating="bad", comment=note_val):
                st.toast("感谢反馈！")
                st.rerun()

    note = st.text_area(
        "补充说明（选填）",
        value=comment,
        placeholder="回答是否准确？有无遗漏或错误？",
        height=72,
        key=note_key,
    )
    if st.button("提交评价", key=f"fb_submit_{idx}", type="primary", use_container_width=True):
        current_rating = meta.get("rating") or ""
        if not current_rating:
            st.toast("请先选择满意度（👍 / 😐 / 👎）")
        elif _apply_feedback(msg, rating=current_rating, comment=note):
            st.toast("评价已保存，感谢！")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _render_message(msg: dict, idx: int) -> None:
    av = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=av):
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            return
        meta = msg.get("meta") or {}
        if meta.get("record_id"):
            _render_feedback(msg, idx)
        if meta.get("model"):
            st.markdown(
                f'<p class="assistant-meta">{meta.get("model", "")} · '
                f'{_fmt_time(meta.get("created_at", ""))}</p>',
                unsafe_allow_html=True,
            )
        _render_sources(msg.get("refs") or [], msg.get("chunks") or [])


def _status_line(status: dict) -> str:
    oer = status.get("oer") or {}
    eo = status.get("eo") or {}
    op, oc = int(oer.get("papers") or 0), int(oer.get("chunks") or 0)
    ep, ec = int(eo.get("papers") or 0), int(eo.get("chunks") or 0)
    cls = "status-ok" if status.get("ok") else "status-bad"
    return (
        f'<span class="{cls}">● OER {op:,} 篇 / {oc:,} 片段 · '
        f"EO {ep:,} 篇 / {ec:,} 片段</span>"
    )


def _render_site_brand(status: dict) -> None:
    if has_full_logo():
        st.markdown(
            f'<div class="site-brand">'
            f'<div class="site-brand-inner site-brand-full">'
            f'<img class="brand-logo-full" src="{brand_logo_full_src()}" alt="电化学大模型"/>'
            f'<p class="brand-status">{_status_line(status)}</p>'
            f"</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="site-brand">'
            f'<div class="site-brand-inner">'
            f'<img class="brand-logo-icon" src="{brand_logo_icon_src()}" alt=""/>'
            f"<div>"
            f'<p class="brand-title">电化学大模型</p>'
            f'<p class="brand-status">{_status_line(status)}</p>'
            f"</div></div></div>",
            unsafe_allow_html=True,
        )


def _render_footer(status: dict) -> None:
    st.markdown(footer_html(index_version=INDEX_VERSION, ok=status["ok"]), unsafe_allow_html=True)


def _render_header(status: dict) -> None:
    _render_site_brand(status)
    c0, c1, c2, c3, c4 = st.columns([0.95, 1.0, 0.72, 0.72, 0.55], gap="small")
    with c0:
        rag_idx = (
            RAG_MODE_OPTIONS.index(st.session_state.rag_mode)
            if st.session_state.rag_mode in RAG_MODE_OPTIONS
            else 0
        )
        st.session_state.rag_mode = st.selectbox(
            "知识库",
            RAG_MODE_OPTIONS,
            format_func=lambda m: RAG_MODE_LABELS.get(m, m),
            index=rag_idx,
            label_visibility="collapsed",
        )
    with c1:
        idx = (
            CHAT_MODEL_OPTIONS.index(st.session_state.chat_model)
            if st.session_state.chat_model in CHAT_MODEL_OPTIONS
            else 0
        )
        st.session_state.chat_model = st.selectbox(
            "模型",
            CHAT_MODEL_OPTIONS,
            index=idx,
            label_visibility="collapsed",
        )
    with c2:
        st.markdown('<div class="hdr-btn">', unsafe_allow_html=True)
        if st.button("新对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.processing_query = None
            st.session_state.page = "chat"
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="hdr-btn">', unsafe_allow_html=True)
        lab = "返回" if st.session_state.page == "history" else "记录"
        if st.button(lab, use_container_width=True):
            st.session_state.page = "history" if st.session_state.page == "chat" else "chat"
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="hdr-btn">', unsafe_allow_html=True)
        with st.popover("⚙ 设置"):
            st.session_state.top_k = st.slider(
                "每次检索文献片段数",
                4,
                12,
                st.session_state.top_k,
                help="越大回答越全面，但可能更慢",
            )
            st.caption(
                f"当前：{RAG_MODE_LABELS.get(st.session_state.rag_mode, '')} 知识库"
            )
            st.divider()
            if st.button("导出提问记录 Excel", use_container_width=True):
                n = export_xlsx(CHAT_LOG_EXPORT_PATH)
                st.caption(f"已导出 {n} 条")
                with CHAT_LOG_EXPORT_PATH.open("rb") as f:
                    st.download_button(
                        "下载 Excel",
                        f.read(),
                        CHAT_LOG_EXPORT_PATH.name,
                        use_container_width=True,
                    )
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="toolbar-spacer" aria-hidden="true"></div>', unsafe_allow_html=True)


def _render_empty(status: dict) -> None:
    papers = int(status.get("papers") or 0)
    chunks = int(status.get("chunks") or 0)
    st.markdown(hero_html(papers=papers, chunks=chunks), unsafe_allow_html=True)
    st.markdown(
        stats_badges_html(
            papers=papers,
            chunks=chunks,
            model=st.session_state.chat_model,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(feature_cards_html(), unsafe_allow_html=True)
    st.markdown('<p class="example-section-title">试试这些示例问题</p>', unsafe_allow_html=True)
    st.markdown('<div class="example-list">', unsafe_allow_html=True)
    for ex in EXAMPLES:
        st.button(
            ex,
            key=f"ex_{ex[:12]}",
            use_container_width=True,
            on_click=_enqueue,
            args=(ex,),
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_history() -> None:
    st.markdown(
        f'<div class="page-banner">'
        f'<img src="{brand_logo_icon_src()}" alt="电化学大模型"/>'
        f"<div><strong>提问记录</strong>"
        f"<span>查看历史问答与用户评价</span></div></div>",
        unsafe_allow_html=True,
    )
    q = st.text_input("搜索", placeholder="过滤问题或回答关键词…", label_visibility="collapsed")
    records = list_records(limit=80)
    if q:
        key = q.lower()
        records = [
            r
            for r in records
            if key in (r.get("question") or "").lower()
            or key in (r.get("answer") or "").lower()
        ]
    if not records:
        st.info("暂无记录")
        return
    for rec in records:
        title = f"{_fmt_time(rec.get('created_at', ''))} · {(rec.get('question') or '')[:42]}"
        with st.expander(title):
            st.markdown("**问**")
            st.write(rec.get("question", ""))
            st.markdown("**答**")
            st.write(rec.get("answer", ""))
            rating = rec.get("rating") or "未评"
            st.caption(f"质量：{RATING_LABELS.get(rating, rating)}")
            if rec.get("comment"):
                st.caption(f"评论：{rec.get('comment')}")
            else:
                st.caption("评论：暂无（可在对话页直接评价）")


def _enqueue(text: str | None = None) -> None:
    # Button on_click passes args=(text,); chat_input on_submit passes nothing —
    # value lives in session_state under the widget key.
    if text is None:
        text = st.session_state.get("main_chat_input") or ""
    text = str(text).strip()
    if not text:
        return
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.processing_query = text
    if "main_chat_input" in st.session_state:
        st.session_state.main_chat_input = ""


def _answer(text: str, top_k: int) -> None:
    status = _index_status()
    answer, refs, chunks = "", [], []
    model = st.session_state.chat_model
    rag_mode = st.session_state.rag_mode

    if not mode_ready(rag_mode, status):
        label = RAG_MODE_LABELS.get(rag_mode, rag_mode)
        answer = f"当前 {label} 知识库暂不可用，请联系管理员完成索引构建。"
    else:
        try:
            get_client()
            prior = _llm_history(st.session_state.messages[:-1])
            result = query_rag(
                text,
                top_k=top_k,
                model=st.session_state.chat_model,
                history=prior,
                rag_mode=rag_mode,
            )
            answer = result["answer"]
            refs = result["refs"]
            chunks = result["chunks"]
            model = result["model"]
        except RuntimeError as exc:
            answer = str(exc)
        except Exception as exc:  # noqa: BLE001
            answer = f"暂时无法回答：{exc}"

    rec = append_record(
        question=text,
        answer=answer,
        model=model,
        top_k=top_k,
        refs=refs,
    )
    meta = {
        "model": model,
        "created_at": rec["created_at"],
        "record_id": rec["id"],
        "rating": "",
        "comment": "",
    }

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "refs": refs, "chunks": chunks, "meta": meta}
    )


# ── bootstrap ──
_page_icon = (
    str(LOGO_ICON_FILE)
    if LOGO_ICON_FILE.is_file()
    else str(LOGO_FULL_FILE)
    if LOGO_FULL_FILE.is_file()
    else "⚗️"
)
st.set_page_config(page_title="电化学大模型", page_icon=_page_icon, layout="wide")
_inject_css()

for key, val in [
    ("messages", []),
    ("chat_model", CHAT_MODEL),
    ("rag_mode", "oer"),
    ("top_k", TOP_K),
    ("page", "chat"),
    ("processing_query", None),
]:
    st.session_state.setdefault(key, val)

status = _index_status()
_render_header(status)

if st.session_state.page == "history":
    _render_history()
    _render_footer(status)
else:
    has_chat = bool(st.session_state.messages or st.session_state.processing_query)
    if has_chat:
        box = st.container(height=480)
        with box:
            for i, m in enumerate(st.session_state.messages):
                _render_message(m, i)
            pending = st.session_state.processing_query
            if pending:
                with st.chat_message("assistant", avatar=BOT_AVATAR):
                    with st.status("正在检索文献并生成回答…", expanded=False):
                        _answer(pending, st.session_state.top_k)
                        st.session_state.processing_query = None
                last_idx = len(st.session_state.messages) - 1
                if last_idx >= 0:
                    _render_message(st.session_state.messages[last_idx], last_idx)
    else:
        _render_empty(status)

    st.chat_input("输入问题，回车发送", on_submit=_enqueue, key="main_chat_input")

    if not st.session_state.processing_query:
        _render_footer(status)
