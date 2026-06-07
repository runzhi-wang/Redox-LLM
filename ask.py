"""Query literature RAG (OER / EO / mixed) with cited answers."""

from __future__ import annotations

import argparse
import sys

from chromadb.errors import NotFoundError

from chroma_utils import create_chroma_client
from config import CHAT_MODEL, TOP_K, get_client
from corpora import RAG_MODE_LABELS, RagMode, corpora_for_mode
from embedder import embed_texts
from citations import strip_answer_preamble, strip_inline_citations
from index_utils import format_cite


SYSTEM_PROMPTS = {
    "oer": """你是电极材料与电催化（OER/HER/电芬顿等）领域的文献助手。
根据提供的 OER 参考资料回答问题，支持多轮追问。规则：
1. 只使用给定资料中的信息；不确定时直接说“文献中未提及”或“未见相关报道”。
2. 直接回答用户问题，禁止以「简要回答」「简短结论」「基于你提供的文献片段」
   「下面归纳片段中」等元叙述开头，也不要交代信息来自哪些片段。
3. 正文保持简洁流畅，不要在段落中插入 [doi #章节] 这类长引用标记；
   来源由界面在回答下方单独展示，你无需手写 DOI 或章节链接。
4. 化学式用纯文本或 $Ru@RuO_2$ 形式，禁止在普通英文/中文句子里使用未转义下划线。
5. 结合对话上下文理解追问，仍须基于本次检索到的资料作答。
6. 用中文回答，条理清晰，像专家直接讲解，不要写汇报式套话。""",
    "eo": """你是电氧化（EO）与有机电合成领域的文献助手。
根据提供的 EO 参考资料回答问题，支持多轮追问。规则：
1. 只使用给定资料中的信息；不确定时直接说“文献中未提及”或“未见相关报道”。
2. 直接回答用户问题，禁止元叙述开头，不要交代信息来自哪些片段。
3. 正文保持简洁流畅，不要在段落中插入 [doi #章节] 这类长引用标记。
4. 化学式用纯文本或 LaTeX 形式，禁止未转义下划线。
5. 结合对话上下文理解追问，仍须基于本次检索到的资料作答。
6. 用中文回答，条理清晰。""",
    "mixed": """你是电化学与电氧化（OER/EO/电芬顿等）领域的文献助手。
根据提供的 OER 与 EO 混合参考资料回答问题，支持多轮追问。规则：
1. 只使用给定资料中的信息；不确定时直接说“文献中未提及”。
2. 直接回答用户问题，禁止元叙述开头。
3. 正文保持简洁流畅，不要在段落中插入 [doi #章节] 这类长引用标记。
4. 化学式用纯文本或 LaTeX 形式。
5. 结合对话上下文理解追问，仍须基于本次检索到的资料作答。
6. 用中文回答，条理清晰。""",
}


def format_context(rows: list[dict]) -> tuple[str, list[dict], list[dict]]:
    lines: list[str] = []
    refs: list[dict] = []
    chunks: list[dict] = []
    for i, row in enumerate(rows, start=1):
        doc = row["text"]
        meta = row["meta"]
        dist = row["distance"]
        chunk_id = row["chunk_id"]
        corpus = row.get("corpus", "")
        doi = meta.get("doi", "")
        section = meta.get("section", "")
        heading_path = meta.get("heading_path", "")
        cite = format_cite(doi, section, heading_path)
        corpus_tag = f" [{corpus.upper()}]" if corpus else ""
        lines.append(
            f"--- 片段 {i}{corpus_tag} {cite} ---\n"
            f"DOI: {doi}\n"
            f"标题路径: {heading_path}\n"
            f"章节: {section}\n"
            f"相似度距离: {dist:.4f}\n"
            f"{doc}\n"
        )
        ref = {
            "cite": cite,
            "chunk_id": chunk_id,
            "doi": doi,
            "section": section,
            "heading_path": heading_path,
            "distance": dist,
            "has_metrics": meta.get("has_metrics", False),
            "corpus": corpus,
        }
        refs.append(ref)
        chunks.append(
            {
                "cite": cite,
                "doi": doi,
                "section": section,
                "heading_path": heading_path,
                "distance": dist,
                "text": doc,
                "has_metrics": meta.get("has_metrics", False),
                "corpus": corpus,
            }
        )
    return "\n".join(lines), refs, chunks


def retrieve(
    question: str,
    top_k: int = TOP_K,
    *,
    rag_mode: RagMode = "oer",
) -> tuple[str, list[dict], list[dict]]:
    corpora = corpora_for_mode(rag_mode)
    chroma = create_chroma_client()
    query_vec = embed_texts([question], is_query=True)[0]
    merged: list[dict] = []

    try:
        for corpus in corpora:
            try:
                col = chroma.get_collection(corpus.collection_name)
            except NotFoundError:
                continue
            if col.count() == 0:
                continue
            results = col.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            for doc, meta, dist, chunk_id in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                merged.append(
                    {
                        "text": doc,
                        "meta": meta,
                        "distance": dist,
                        "chunk_id": chunk_id,
                        "corpus": corpus.key,
                    }
                )
    finally:
        try:
            chroma.close()
        except Exception:  # noqa: BLE001
            pass

    merged.sort(key=lambda r: r["distance"])
    merged = merged[:top_k]
    if not merged:
        return "", [], []
    return format_context(merged)


def _build_chat_messages(
    question: str,
    context: str,
    history: list[dict] | None,
    *,
    rag_mode: RagMode,
) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPTS.get(rag_mode, SYSTEM_PROMPTS["oer"])}
    ]
    if history:
        for msg in history:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": f"问题：{question}\n\n参考资料：\n{context}",
        }
    )
    return messages


def query_rag(
    question: str,
    top_k: int = TOP_K,
    *,
    model: str | None = None,
    history: list[dict] | None = None,
    rag_mode: RagMode = "oer",
) -> dict:
    chat_model = model or CHAT_MODEL
    context, refs, chunks = retrieve(question, top_k=top_k, rag_mode=rag_mode)
    if not context:
        label = RAG_MODE_LABELS.get(rag_mode, rag_mode)
        return {
            "question": question,
            "answer": f"当前 {label} 知识库暂无可用片段，请联系管理员完成索引构建。",
            "raw_answer": "",
            "refs": [],
            "chunks": [],
            "top_k": top_k,
            "model": chat_model,
            "rag_mode": rag_mode,
        }

    client = get_client()
    messages = _build_chat_messages(question, context, history, rag_mode=rag_mode)

    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0.2,
    )
    raw_answer = response.choices[0].message.content or ""
    answer = strip_answer_preamble(strip_inline_citations(raw_answer))
    return {
        "question": question,
        "answer": answer,
        "raw_answer": raw_answer,
        "refs": refs,
        "chunks": chunks,
        "top_k": top_k,
        "model": chat_model,
        "rag_mode": rag_mode,
    }


def ask(question: str, top_k: int = TOP_K, *, rag_mode: RagMode = "oer") -> str:
    result = query_rag(question, top_k=top_k, rag_mode=rag_mode)
    ref_lines = "\n".join(
        f"  - {r['cite']}  {r.get('heading_path', '')}  (dist={r['distance']:.4f})"
        for r in result["refs"]
    )
    return f"{result['answer']}\n\n--- 检索来源 ---\n{ref_lines}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Query literature RAG")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help="Top-k chunks")
    parser.add_argument(
        "--mode",
        choices=["oer", "eo", "mixed"],
        default="oer",
        help="RAG corpus mode",
    )
    args = parser.parse_args()

    if not args.question:
        args.question = input("请输入问题: ").strip()
    if not args.question:
        print("未提供问题。")
        return 1

    print(ask(args.question, top_k=args.top_k, rag_mode=args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
