"""3단계(검색). 요약으로 검색하고 원본을 돌려주는 MultiVectorRetriever.

- vectorstore(Chroma) : 요약 임베딩 (+ 텍스트·표는 원문 임베딩도) → 검색 대상
- docstore(InMemory)  : 원본(텍스트, 표 HTML, 이미지 base64) → 답변 재료
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
