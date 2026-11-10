"""1단계. PDF에서 텍스트·표·이미지를 추출하고 텍스트를 분할합니다.

수업 노트북 '01 데이터 추출 및 분할'의 함수를 그대로 따르고,
답변 근거를 표시할 수 있도록 페이지 번호와 표 HTML만 함께 보관합니다.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(pdf_path: str | Path, figures_dir: str | Path):
    """PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다. (수업 01 [3])"""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    # 이전 실행에서 남은 그림이 섞이지 않도록 지웁니다.
    for old in figures_dir.glob("*.jpg"):
        old.unlink()

    return partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",                 # 레이아웃 모델로 표·그림 영역까지 인식
        extract_images_in_pdf=True,        # 그림을 jpg 파일로 저장
        infer_table_structure=True,        # 표의 행·열 구조를 HTML로 보존
        chunking_strategy="by_title",      # 타이틀 기준으로 텍스트 분할
        max_characters=4000,               # 최대 4000자로 블록 제한
        new_after_n_chars=3800,            # 3800자 이후 새 블록 생성
        combine_text_under_n_chars=2000,   # 2000자 이하는 결합
        extract_image_block_output_dir=str(figures_dir),
    )


def categorize_elements(raw_pdf_elements) -> tuple[list[dict], list[dict]]:
    """추출 요소를 텍스트와 표로 분류합니다. (수업 01 [4])

    수업 코드는 str(element)만 담았지만, 여기서는 근거 표시용 페이지 번호와
    표의 HTML(text_as_html)을 함께 담습니다.
    """
    texts: list[dict] = []
    tables: list[dict] = []
    for element in raw_pdf_elements:
        page = element.metadata.page_number
        if "unstructured.documents.elements.Table" in str(type(element)):
            html = getattr(element.metadata, "text_as_html", None)
            tables.append({"kind": "table", "content": html or str(element), "page": page})
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append({"kind": "text", "content": str(element), "page": page})
    return texts, tables


def list_figures(figures_dir: str | Path) -> list[dict]:
    """figures/ 에 저장된 그림(figure-<페이지>-<번호>.jpg)을 페이지 순으로 돌려줍니다."""
    items = []
    for path in Path(figures_dir).glob("figure-*.jpg"):
        _, page, number = path.stem.split("-")
        items.append({"kind": "image", "content": str(path), "page": int(page),
                      "number": int(number), "image_path": str(path)})
    return sorted(items, key=lambda x: (x["page"], x["number"]))


def save_pickle(obj, path: str | Path) -> None:
    """추출·요약 결과를 저장합니다. 재실행 시간과 API 비용을 아끼기 위함입니다. (수업 01 [10])"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)
