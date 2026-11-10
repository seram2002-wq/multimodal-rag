"""3단계. 요약으로 검색하고 원본으로 답하는 다중 벡터 검색기와 RAG 체인.

- vectorstore(Chroma) : 요약 임베딩 (+ 텍스트·표는 원문 임베딩도) → 검색 대상
- docstore(InMemory)  : 원본(텍스트, 표 HTML, 이미지 base64) → 답변 재료
두 저장소는 doc_id 하나로 연결됩니다.

질문 처리는 FashionRAG 수업의 파이프라인 구조를 따릅니다.
 ① 번역 체인 (한→영) ② 검색 ③ 프롬프트 구성 ④ RAG 체인 (비전) ⑤ 번역 체인 (영→한)

단독 실행 (프로젝트 루트에서):
    python -m src.retriever            # 질의응답 5건
    python -m src.retriever --before   # + Before(텍스트 RAG) 비교
"""
from __future__ import annotations

import os
import sys
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.extract import load_or_extract, split_texts
from src.summarize import encode_image, load_or_summarize

ID_KEY = "doc_id"


def _embeddings(embeddings: Embeddings | None) -> Embeddings:
    return embeddings or OpenAIEmbeddings(model="text-embedding-3-small")


def create_multi_vector_retriever(
    pairs: list[tuple[list[dict], list[str]]],
    *,
    k: int = 6,
    embeddings: Embeddings | None = None,
    index_originals: bool = True,
) -> MultiVectorRetriever:
    """(원본 리스트, 요약 리스트) 쌍을 받아 검색기를 만듭니다.

    예) [(texts, text_summaries), (tables, table_summaries), (images, image_summaries)]

    index_originals=True 이면 텍스트·표는 요약과 함께 '원문'도 같은 doc_id 로 임베딩합니다.
    요약에서 빠진 세부 수치도 검색되게 하기 위함입니다. (하나의 원본 = 여러 벡터)
    """
    vectorstore = Chroma(
        collection_name=f"mm-rag-{uuid4().hex[:8]}",   # 셀을 다시 실행해도 중복 저장되지 않게
        embedding_function=_embeddings(embeddings),
    )
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        id_key=ID_KEY,
        search_kwargs={"k": k},
    )

    for items, summaries in pairs:
        if len(items) != len(summaries):
            raise ValueError("원본 개수와 요약 개수가 다릅니다.")
        if not items:
            continue
        doc_ids = [str(uuid4()) for _ in items]
        summary_docs, originals = [], []
        for doc_id, item, summary in zip(doc_ids, items, summaries):
            meta = {"kind": item["kind"], "page": item["page"]}
            summary_docs.append(Document(page_content=summary, metadata={ID_KEY: doc_id, **meta}))
            if item["kind"] == "image":
                # 그림 원본(base64)과 함께 요약도 보관해 답변 단계에서 캡션 정보를 쓸 수 있게 함
                meta["image_path"] = item["image_path"]
                meta["summary"] = summary
                content = encode_image(item["image_path"])
            else:
                content = item["content"]
                if index_originals:
                    summary_docs.append(Document(page_content=content,
                                                 metadata={ID_KEY: doc_id, **meta}))
            originals.append(Document(page_content=content, metadata=meta))
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, originals)))

    return retriever


def create_text_retriever(
    chunks: list[str],
    *,
    k: int = 6,
    embeddings: Embeddings | None = None,
):
    """Before 비교용: 텍스트 청크 원문만 그대로 임베딩하는 일반 텍스트 RAG 검색기.

    표·그림, 요약을 쓰지 않습니다.
    """
    docs = [Document(page_content=c, metadata={"kind": "text"}) for c in chunks]
    vectorstore = Chroma.from_documents(
        docs,
        embedding=_embeddings(embeddings),
        collection_name=f"text-rag-{uuid4().hex[:8]}",
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ============================================================
# 체인 (FashionRAG: 프롬프트 | 모델 | 파서)
# ============================================================
QUESTIONS = [
    "CIFAR-10 unconditional 설정에서 NCSN의 Inception score와 FID는 각각 얼마인가요?",
    "Table 1에서 Inception score가 NCSN보다 높은 모델은 무엇이고, 어떤 설정인가요?",
    "Figure 2에서 빨간 사각형으로 표시된 영역은 무엇을 의미하나요?",
    "Figure 3에서 annealed Langevin dynamics와 일반 Langevin dynamics의 결과는 어떻게 다른가요?",
    "이미지 생성 실험에서 노이즈 수준 σ는 몇 개, 어떤 범위로 설정했나요?",
]

RAG_SYSTEM_PROMPT = (
    "You are an assistant answering questions about a machine learning research paper. "
    "Answer only from the provided context: text passages, tables (HTML) and figures. "
    "Keep numbers and model names exactly as written. "
    "The text was extracted from a PDF, so symbols may be broken by OCR "
    "(e.g. 'σ' may appear as 'o' or '0', subscripts may be flattened). "
    "For figures, use both the attached image and its description. "
    "Tables are given as HTML: read every row, including group headers such as settings, "
    "and compare the values yourself when the question asks for a comparison. "
    "If the context is not enough, say \"The provided context does not contain the answer.\""
)


def _chat(model: str | None = None) -> ChatOpenAI:
    load_dotenv()
    # temperature=0 : 번역·답변처럼 '정답에 가까운 출력'이 필요할 때 무작위성 최소화
    return ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.0)


def create_translate_chain(model: str | None = None):
    """{"text", "target_lang"} → 번역 문자열 (FashionRAG [12])."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following text into {target_lang}. "
                   "Keep technical terms, model names and numbers as they are. "
                   "Output only the translation."),
        ("user", "{text}"),
    ])
    return prompt | _chat(model) | StrOutputParser()


def create_rag_chain(model: str | None = None):
    """{"user_query", "context", "image_messages"} → 답변 문자열.

    image_messages : 검색된 그림을 data URL 이미지 블록으로 담은 메시지 (없으면 비어 있음)
    질문은 그림 '뒤' 마지막 메시지에 둡니다. 그림이 마지막에 오면 모델이 질문 대신
    그림 설명만 하는 경우가 있었기 때문입니다.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("user", "Context:\n{context}"),
        MessagesPlaceholder("image_messages", optional=True),
        ("user", "Question: {user_query}\n"
                 "Answer this question using the context and figures above. "
                 "Do not just describe the figures."),
    ])
    return prompt | _chat(model) | StrOutputParser()


def format_prompt_inputs(docs: list[Document], user_query: str) -> dict:
    """검색된 원본 문서와 질문으로 RAG 프롬프트 입력 dict 를 만듭니다. (FashionRAG [13])"""
    context_parts, image_blocks = [], []
    for d in docs:
        kind, page = d.metadata["kind"], d.metadata.get("page")
        tag = f"[{kind} p.{page}]" if page else f"[{kind}]"
        if kind == "image":
            context_parts.append(f"{tag} (attached figure) description: {d.metadata.get('summary', '')}")
            image_blocks.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{d.page_content}"}})
        else:
            context_parts.append(f"{tag}\n{d.page_content}")
    return {
        "user_query": user_query,
        "context": "\n\n".join(context_parts),
        "image_messages": [HumanMessage(content=image_blocks)] if image_blocks else [],
    }


def sources(docs: list[Document]) -> str:
    """'table p.7, image p.4' 형태로 근거 원본의 종류와 페이지를 적습니다."""
    return ", ".join(f"{d.metadata['kind']} p.{d.metadata['page']}" if d.metadata.get("page")
                     else d.metadata["kind"] for d in docs)


def ask_paper_rag(query_ko: str, retriever, *, translate_chain, rag_chain) -> dict:
    """한국어 질문 하나로 ①~⑤를 실행하고 결과 dict 를 돌려줍니다. (FashionRAG [14])"""
    query_en = translate_chain.invoke({"text": query_ko, "target_lang": "English"})      # ①
    docs = retriever.invoke(query_en)                                                     # ②
    answer_en = rag_chain.invoke(format_prompt_inputs(docs, query_en))                    # ③④
    answer_ko = translate_chain.invoke({"text": answer_en, "target_lang": "Korean"})      # ⑤
    return {"query_ko": query_ko, "query_en": query_en, "docs": docs,
            "answer_en": answer_en, "answer_ko": answer_ko, "sources": sources(docs)}


def build_retriever(data: dict, summaries: dict, *, k: int = 6) -> MultiVectorRetriever:
    """추출 결과와 요약으로 다중 벡터 검색기를 만듭니다."""
    return create_multi_vector_retriever(
        [(data["texts"], summaries["texts"]),
         (data["tables"], summaries["tables"]),
         (data["images"], summaries["images"])],
        k=k,
    )


if __name__ == "__main__":
    data = load_or_extract()                 # 저장본이 없으면 추출
    summaries = load_or_summarize(data)      # 저장본이 없으면 요약
    translate_chain, rag_chain = create_translate_chain(), create_rag_chain()

    mm_retriever = build_retriever(data, summaries)
    mm_results = []
    for i, q in enumerate(QUESTIONS, start=1):
        r = ask_paper_rag(q, mm_retriever, translate_chain=translate_chain, rag_chain=rag_chain)
        mm_results.append(r)
        print(f"\n{'=' * 30} Q{i} {'=' * 30}")
        print("질문        :", r["query_ko"])
        print("번역된 질문 :", r["query_en"])
        print("근거 원본   :", r["sources"])
        print(r["answer_ko"])

    if "--before" in sys.argv:
        text_retriever = create_text_retriever(split_texts(data["texts"]), k=6)
        for i, (q, after) in enumerate(zip(QUESTIONS, mm_results), start=1):
            before = ask_paper_rag(q, text_retriever, translate_chain=translate_chain, rag_chain=rag_chain)
            print(f"\n{'=' * 30} Q{i} Before / After {'=' * 30}")
            print("질문:", q)
            print(f"[Before 텍스트 RAG]   근거: {before['sources']}\n{before['answer_ko']}")
            print(f"[After 멀티모달 RAG]  근거: {after['sources']}\n{after['answer_ko']}")
