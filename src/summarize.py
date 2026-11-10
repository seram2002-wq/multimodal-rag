"""2단계. 텍스트·표·이미지를 검색용 요약으로 바꿉니다.

FashionRAG 수업의 LCEL(프롬프트 | 모델 | 파서) 구조와
base64 data URL 로 이미지를 비전 모델에 보내는 방법을 그대로 사용합니다.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 논문 원문이 영어이므로 요약도 영어로 만듭니다.
# (질문은 노트북의 번역 체인으로 영어로 바꿔 검색합니다)
# 요약이 검색 색인이 되므로, 수치·설정값이 빠지면 그 내용은 검색되지 않습니다.
# 그래서 길이보다 '세부 정보 보존'을 우선하도록 지시합니다.
TEXT_PROMPT = (
    "You are an assistant tasked with summarizing text from a research paper for retrieval. "
    "Write a detailed summary that keeps every number, hyperparameter, experimental setting, "
    "dataset name, model name and figure/table reference exactly as written. "
    "If the text contains figure or table captions, restate them. "
    "Do not add information that is not in the source.\n\ntext: {element}"
)

TABLE_PROMPT = (
    "You are an assistant tasked with describing a table (HTML) from a research paper for retrieval. "
    "First say what the table compares. Then list every row with the values of ALL columns "
    "(e.g. 'ModelA: Inception 8.87, FID 25.32'), keeping group headers such as settings or categories. "
    "Do not drop any column or row.\n\ntable: {element}"
)

IMAGE_PROMPT = (
    "You are an assistant tasked with summarizing figures from a research paper for retrieval. "
    "Describe the figure in 3-5 sentences. "
    "If it is a plot, describe the axes, legend and trend. "
    "If it is a grid of generated samples, say what kind of images they are. "
    "Copy any visible text or numbers exactly. Do not guess what is not visible.\n"
    "Below is the text from the same page, which may contain this figure's caption. "
    "If a caption matches this figure, include the figure number and explain what its marks "
    "(colors, boxes, panels) mean according to the caption.\n\npage text: {context}"
)


def _model(model: str | None = None) -> ChatOpenAI:
    load_dotenv()
    # temperature=0 : 요약은 '정답에 가까운 출력'이 필요하므로 무작위성 최소화
    return ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


def encode_image(path: str | Path) -> str:
    """이미지 파일 → base64 문자열 (FashionRAG [13])."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def guess_mime(path: str | Path) -> str:
    """확장자로 MIME 타입 추정 (FashionRAG [4])."""
    return "image/png" if str(path).lower().endswith(".png") else "image/jpeg"


def summarize_elements(items: list[dict], *, model: str | None = None) -> list[str]:
    """텍스트 또는 표 항목을 요약합니다. 입력 순서와 같은 순서로 돌려줍니다."""
    if not items:
        return []
    template = TABLE_PROMPT if items[0]["kind"] == "table" else TEXT_PROMPT
    chain = ChatPromptTemplate.from_template(template) | _model(model) | StrOutputParser()
    results = chain.batch([{"element": it["content"]} for it in items], {"max_concurrency": 5})
    return [str(r) for r in results]   # 순수 문자열로 저장 (pickle 호환)


def summarize_images(items: list[dict], contexts: list[str] | None = None, *,
                     model: str | None = None) -> list[str]:
    """이미지 항목을 비전 모델로 요약합니다. 입력 순서와 같은 순서로 돌려줍니다.

    contexts : 각 그림과 같은 페이지의 본문 (그림 캡션을 함께 읽게 하기 위함)
    """
    if not items:
        return []
    contexts = contexts or [""] * len(items)
    prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {"type": "text", "text": IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": "data:{image_mime};base64,{image_data}"}},
        ]),
    ])
    chain = prompt | _model(model) | StrOutputParser()
    inputs = [{"image_mime": guess_mime(it["image_path"]),
               "image_data": encode_image(it["image_path"]),
               "context": ctx[:3000]} for it, ctx in zip(items, contexts)]
    return [str(s) for s in chain.batch(inputs, {"max_concurrency": 5})]
