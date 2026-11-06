"""2단계. 텍스트·표·이미지를 검색용 요약으로 바꿉니다.

요약은 '검색에 걸리기 위한 색인'이고, 답변은 3단계에서 원본으로 만듭니다.
- 텍스트·표 : gpt-4o-mini 텍스트 입력
- 이미지    : gpt-4o-mini 비전 입력 (base64)
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 질문을 한국어로 할 것이므로 요약도 한국어로 만들고,
# 영어 논문의 수치·모델명·데이터셋명은 원문 표기 그대로 두게 합니다.
TEXT_TABLE_PROMPT = """당신은 영어 논문을 검색용으로 요약하는 조수입니다.
아래 {kind}을(를) 한국어 3~5문장으로 요약하세요.
- 수치, 모델명, 데이터셋명, 수식 기호는 원문 표기 그대로 유지합니다.
- 표라면 어떤 항목을 비교하는 표인지와 핵심 수치를 적습니다.
- 원문에 없는 내용은 추가하지 않습니다.

{kind}:
{content}"""

IMAGE_PROMPT = """당신은 영어 논문의 그림을 검색용으로 요약하는 조수입니다.
이 그림을 한국어 3~5문장으로 요약하세요.
- 그래프라면 축, 범례, 추세를 적습니다.
- 생성 샘플·이미지 격자라면 무엇을 보여주는지 적습니다.
- 그림 안의 글자와 수치는 원문 표기 그대로 적습니다.
- 보이지 않는 내용은 추측하지 않습니다."""

KIND_KO = {"text": "본문", "table": "표(HTML)"}


def _llm(model: str | None) -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


def summarize_texts(items: list[dict], *, model: str | None = None) -> list[str]:
    """텍스트 또는 표 항목 리스트를 요약 문자열 리스트로 돌려줍니다 (순서 동일)."""
    if not items:
        return []
    chain = ChatPromptTemplate.from_template(TEXT_TABLE_PROMPT) | _llm(model) | StrOutputParser()
    inputs = [{"kind": KIND_KO[item["kind"]], "content": item["content"]} for item in items]
    return chain.batch(inputs, {"max_concurrency": 5})


def encode_image(image_path: str | Path) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def summarize_images(items: list[dict], *, model: str | None = None) -> list[str]:
    """이미지 항목 리스트를 요약 문자열 리스트로 돌려줍니다 (순서 동일)."""
    if not items:
        return []
    llm = _llm(model)
    messages = [
        [HumanMessage(content=[
            {"type": "text", "text": IMAGE_PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{encode_image(item['image_path'])}"}},
        ])]
        for item in items
    ]
    responses = llm.batch(messages, {"max_concurrency": 5})
    return [str(response.content) for response in responses]
