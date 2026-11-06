"""3단계(검색). 요약으로 검색하고 원본을 돌려주는 MultiVectorRetriever.

- vectorstore(Chroma) : 요약 임베딩만 저장  → 검색 대상
- docstore(InMemory)  : 원본(텍스트, 표 HTML, 이미지 base64) 저장 → 답변 재료
두 저장소는 doc_id 하나로 연결됩니다.
"""
from __future__ import annotations

from uuid import uuid4

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.stores import InMemoryStore
from langchain_openai import OpenAIEmbeddings

from src.summarize import encode_image

ID_KEY = "doc_id"


def _embeddings(embeddings: Embeddings | None) -> Embeddings:
    return embeddings or OpenAIEmbeddings(model="text-embedding-3-small")


def build_multi_vector_retriever(
    pairs: list[tuple[list[dict], list[str]]],
    *,
    k: int = 4,
    embeddings: Embeddings | None = None,
) -> MultiVectorRetriever:
    """(원본 항목 리스트, 요약 리스트) 쌍들을 받아 검색기를 만듭니다.

    예) [(texts, text_summaries), (tables, table_summaries), (images, image_summaries)]
    """
    vectorstore = Chroma(
        collection_name=f"multimodal-{uuid4().hex[:8]}",  # 셀을 다시 실행해도 중복 저장되지 않게
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
        ids = [str(uuid4()) for _ in items]
        summary_docs = []
        originals = []
        for doc_id, item, summary in zip(ids, items, summaries):
            meta = {"kind": item["kind"], "page": item["page"], "source": item["source"]}
            summary_docs.append(Document(page_content=summary, metadata={ID_KEY: doc_id, **meta}))
            if item["kind"] == "image":
                content = encode_image(item["image_path"])
                meta["image_path"] = item["image_path"]
            else:
                content = item["content"]
            originals.append(Document(page_content=content, metadata=meta))
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(ids, originals)))

    return retriever


def build_text_retriever(
    texts: list[dict],
    *,
    k: int = 4,
    embeddings: Embeddings | None = None,
):
    """Before 비교용: 텍스트 청크 원문만 그대로 임베딩하는 일반 텍스트 RAG 검색기."""
    docs = [
        Document(page_content=item["content"],
                 metadata={"kind": "text", "page": item["page"], "source": item["source"]})
        for item in texts
    ]
    vectorstore = Chroma.from_documents(
        docs,
        embedding=_embeddings(embeddings),
        collection_name=f"text-only-{uuid4().hex[:8]}",
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
