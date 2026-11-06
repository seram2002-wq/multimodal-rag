"""3단계(생성). 검색된 원본(텍스트·표·이미지)을 LLM에 함께 넣어 답변합니다."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """당신은 영어 논문에 대한 질문에 답하는 조수입니다.
제공된 본문, 표(HTML), 그림만 근거로 한국어로 답하세요.
- 수치와 모델명은 원문 표기 그대로 씁니다.
- 근거가 부족하면 "제공된 자료에서 찾을 수 없습니다"라고 답합니다.
- 답변 마지막 줄에 사용한 근거를 [종류 p.페이지] 형식으로 적습니다."""


def _to_messages(inputs: dict) -> list:
    docs: list[Document] = inputs["docs"]
    blocks = []
    image_parts = []
    for doc in docs:
        kind, page = doc.metadata.get("kind"), doc.metadata.get("page")
        if kind == "image":
            blocks.append(f"[image p.{page}] (아래 첨부 그림)")
            image_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{doc.page_content}"},
            })
        else:
            blocks.append(f"[{kind} p.{page}]\n{doc.page_content}")

    text = "질문: {q}\n\n참고 자료:\n{ctx}".format(q=inputs["question"], ctx="\n\n".join(blocks))
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=[{"type": "text", "text": text}, *image_parts]),
    ]


def build_rag_chain(retriever, *, llm: BaseChatModel | None = None, model: str | None = None):
    """질문 문자열을 받아 {"question", "docs", "answer"}를 돌려주는 체인."""
    if llm is None:
        load_dotenv()
        llm = ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    return RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(
        answer=RunnableLambda(_to_messages) | llm | StrOutputParser()
    )


def sources(docs: list[Document]) -> str:
    """검색된 원본의 종류와 페이지를 'table p.7, image p.8' 형태로 요약합니다."""
    return ", ".join(f"{d.metadata.get('kind')} p.{d.metadata.get('page')}" for d in docs)
