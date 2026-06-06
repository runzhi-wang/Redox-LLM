"""Query OER literature RAG with cited answers."""

from __future__ import annotations

import argparse
import sys

from chroma_utils import ensure_collection_readable
from config import CHAT_MODEL, TOP_K, get_client
from embedder import embed_texts
from citations import strip_answer_preamble, strip_inline_citations
from index_utils import format_cite


SYSTEM_PROMPT = """你是电极材料与电催化（OER/HER/电芬顿等）领域的文献助手。
根据提供的参考资料回答问题，支持多轮追问。规则：
1. 只使用给定资料中的信息；不确定时直接说“文献中未提及”或“未见相关报道”。
2. 直接回答用户问题，禁止以「简要回答」「简短结论」「基于你提供的文献片段」
   「下面归纳片段中」等元叙述开头，也不要交代信息来自哪些片段。
3. 正文保持简洁流畅，不要在段落中插入 [doi #章节] 这类长引用标记；
   来源由界面在回答下方单独展示，你无需手写 DOI 或章节链接。
4. 化学式用纯文本或 $Ru@RuO_2$ 形式，禁止在普通英文/中文句子里使用未转义下划线。
5. 结合对话上下文理解追问，仍须基于本次检索到的资料作答。
6. 用中文回答，条理清晰，像专家直接讲解，不要写汇报式套话。"""


def format_context(results) -> tuple[str, list[dict]]:
    lines: list[str] = []
    refs: list[dict] = []
    for i, (doc, meta, dist, chunk_id) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ),
        start=1,
    ):
        doi = meta.get("doi", "")
        section = meta.get("section", "")
        heading_path = meta.get("heading_path", "")
        cite = format_cite(doi, section, heading_path)
        lines.append(
            f"--- 片段 {i} {cite} ---\n"
            f"DOI: {doi}\n"
            f"标题路径: {heading_path}\n"
            f"章节: {section}\n"
            f"相似度距离: {dist:.4f}\n"
            f"{doc}\n"
        )
        refs.append(
            {
                "cite": cite,
                "chunk_id": chunk_id,
                "doi": doi,
                "section": section,
                "heading_path": heading_path,
                "distance": dist,
                "has_metrics": meta.get("has_metrics", False),
            }
        )
    return "\n".join(lines), refs


def retrieve(question: str, top_k: int = TOP_K) -> tuple[str, list[dict], list[dict]]:
    chroma, collection = ensure_collection_readable()
    try:
        query_vec = embed_texts([question], is_query=True)[0]

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        context, refs = format_context(results)
        chunks = [
            {
                "cite": refs[i]["cite"],
                "doi": refs[i]["doi"],
                "section": refs[i]["section"],
                "heading_path": refs[i]["heading_path"],
                "distance": refs[i]["distance"],
                "text": results["documents"][0][i],
                "has_metrics": refs[i].get("has_metrics", False),
            }
            for i in range(len(refs))
        ]
        return context, refs, chunks
    finally:
        try:
            chroma.close()
        except Exception:  # noqa: BLE001
            pass


def _build_chat_messages(
    question: str,
    context: str,
    history: list[dict] | None,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
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
) -> dict:
    chat_model = model or CHAT_MODEL
    context, refs, chunks = retrieve(question, top_k=top_k)
    client = get_client()
    messages = _build_chat_messages(question, context, history)

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
    }


def ask(question: str, top_k: int = TOP_K) -> str:
    result = query_rag(question, top_k=top_k)
    ref_lines = "\n".join(
        f"  - {r['cite']}  {r.get('heading_path', '')}  (dist={r['distance']:.4f})"
        for r in result["refs"]
    )
    return f"{result['answer']}\n\n--- 检索来源 ---\n{ref_lines}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Query OER literature RAG")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help="Top-k chunks")
    args = parser.parse_args()

    if not args.question:
        args.question = input("请输入问题: ").strip()
    if not args.question:
        print("未提供问题。")
        return 1

    print(ask(args.question, top_k=args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
