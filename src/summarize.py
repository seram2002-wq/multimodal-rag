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
# (질문은 번역 체인으로 영어로 바꿔 검색합니다 — src/chain.py)
TEXT_TABLE_PROMPT = (
    "You are an assistant tasked with summarizing {kind} from a research paper "
    "for retrieval. Give a concise summary in 3-5 sentences. "
    "Keep numbers, model names, dataset names and figure/table numbers exactly as written. "
    "Do not add information that is not in the source.\n\n{kind}: {element}"
)

IMAGE_PROMPT = (
    "You are an assistant tasked with summarizing figures from a research paper "
    "for retrieval. Describe the figure in 3-5 sentences. "
    "If it is a plot, describe the axes, legend and trend. "
    "If it is a grid of generated samples, say what kind of images they are. "
    "Copy any visible text or numbers exactly. Do not guess what is not visible."
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
    prompt = ChatPromptTemplate.from_template(TEXT_TABLE_PROMPT)
    chain = prompt | _model(model) | StrOutputParser()
    kind = "tables (HTML)" if items[0]["kind"] == "table" else "text"
    return chain.batch([{"kind": kind, "element": it["content"]} for it in items],
                       {"max_concurrency": 5})


def summarize_images(items: list[dict], *, model: str | None = None) -> list[str]:
    """이미지 항목을 비전 모델로 요약합니다. 입력 순서와 같은 순서로 돌려줍니다."""
    if not items:
        return []
    prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {"type": "text", "text": IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": "data:{image_mime};base64,{image_data}"}},
        ]),
    ])
    chain = prompt | _model(model) | StrOutputParser()
    inputs = [{"image_mime": guess_mime(it["image_path"]),
               "image_data": encode_image(it["image_path"])} for it in items]
    return chain.batch(inputs, {"max_concurrency": 5})
