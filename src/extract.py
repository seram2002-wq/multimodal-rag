"""1단계. PDF에서 텍스트·표·이미지를 추출합니다.

수업(step1_parse.py)과 같은 Unstructured `partition` 방식을 쓰되,
멀티모달 RAG에 필요한 두 가지 옵션을 더합니다.
- extract_image_block_types : 그림을 잘라 figures/ 에 jpg로 저장
- chunking_strategy="by_title" : 제목(섹션) 단위로 텍스트를 묶음 (step2_langchain.py와 동일)
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from unstructured.partition.pdf import partition_pdf


def extract_pdf(
    pdf_path: str | Path,
    figures_dir: str | Path = "figures",
    *,
    max_characters: int = 1500,
    new_after_n_chars: int = 1200,
    combine_text_under_n_chars: int = 200,
    languages: tuple[str, ...] = ("eng",),
) -> dict[str, list[dict]]:
    """PDF를 hi_res로 파싱해 texts / tables / images 세 묶음으로 돌려줍니다.

    각 항목은 {"kind", "content", "page", "source"} 딕셔너리이며,
    이미지 항목에는 "image_path"가 추가됩니다.
    """
    pdf = Path(pdf_path)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF를 찾을 수 없습니다: {pdf}")

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 이전 실행에서 남은 그림이 섞이지 않도록 지웁니다.
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    elements = partition_pdf(
        filename=str(pdf),
        strategy="hi_res",                      # 표·그림 영역을 인식하려면 hi_res 필요
        languages=list(languages),
        infer_table_structure=True,             # 표를 HTML 구조로 보존
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=str(out_dir),
        chunking_strategy="by_title",
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine_text_under_n_chars,
    )

    texts: list[dict] = []
    tables: list[dict] = []
    for el in elements:
        base = {"page": el.metadata.page_number, "source": pdf.name}
        if el.category in ("Table", "TableChunk"):
            html = getattr(el.metadata, "text_as_html", None)
            tables.append({"kind": "table", "content": html or el.text, **base})
        elif el.text.strip():
            texts.append({"kind": "text", "content": el.text, **base})

    # Unstructured는 그림을 figure-<페이지>-<번호>.jpg, 표 영역을 table-<페이지>-<번호>.jpg 로 저장합니다.
    # 표는 위의 HTML로 다루므로, 이미지 요약 대상은 figure-* 만 사용합니다.
    images: list[dict] = []
    for path in sorted(out_dir.glob("figure-*.jpg"), key=_figure_sort_key):
        page = int(path.stem.split("-")[1])
        images.append({
            "kind": "image",
            "content": str(path),
            "page": page,
            "source": pdf.name,
            "image_path": str(path),
        })

    return {"texts": texts, "tables": tables, "images": images, "categories": _count(elements)}


def _figure_sort_key(path: Path) -> tuple[int, int]:
    nums = [int(n) for n in re.findall(r"\d+", path.stem)]
    return (nums + [0, 0])[0], (nums + [0, 0])[1]


def _count(elements) -> dict[str, int]:
    return dict(Counter(el.category for el in elements).most_common())
