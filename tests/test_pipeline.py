from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pymupdf

from src.extract import extract_pdf
from src.retriever import search
from src.summarize import summarize_items


class ExtractorTests(unittest.TestCase):
    def test_extracts_text_table_and_image_with_page_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "fixture.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((50, 50), "Revenue increased by 12 percent.")

            values = ("Year", "Growth", "2025", "12%", "Total", "12%")
            for row in range(3):
                for column in range(2):
                    rectangle = pymupdf.Rect(
                        50 + column * 120,
                        100 + row * 30,
                        50 + (column + 1) * 120,
                        100 + (row + 1) * 30,
                    )
                    page.draw_rect(rectangle)
                    page.insert_text(
                        (rectangle.x0 + 5, rectangle.y0 + 18),
                        values[row * 2 + column],
                    )

            source_image = temp_path / "source.png"
            pixmap = pymupdf.Pixmap(
                pymupdf.csRGB, pymupdf.IRect(0, 0, 12, 12), 0
            )
            pixmap.clear_with(0x336699)
            pixmap.save(source_image)
            page.insert_image(pymupdf.Rect(350, 50, 380, 80), filename=str(source_image))
            document.save(pdf_path)
            document.close()

            items = extract_pdf(pdf_path, temp_path / "figures")

            self.assertTrue(any(item["kind"] == "text" for item in items))
            self.assertTrue(any(item["kind"] == "table" for item in items))
            images = [item for item in items if item["kind"] == "image"]
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0]["page"], 1)
            self.assertTrue(Path(images[0]["image_path"]).is_file())


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.items = [
            {
                "id": "table-1",
                "source": "report.pdf",
                "page": 4,
                "kind": "table",
                "summary": "2025년 1분기 매출은 12% 증가했다.",
                "original": "기간 | 매출 성장률\n2025년 1분기 | 12%",
            },
            {
                "id": "text-1",
                "source": "report.pdf",
                "page": 7,
                "kind": "text",
                "summary": "신규 제품 출시와 해외 시장 확대가 핵심 전략이다.",
                "original": "회사는 신규 제품을 출시할 계획이다.",
            },
        ]

    def test_search_ranks_summary_and_returns_original(self) -> None:
        results = search("2025년 1분기 매출", self.items, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "table-1")
        self.assertEqual(results[0]["original"], self.items[0]["original"])
        self.assertGreater(results[0]["score"], 0)

    def test_search_returns_empty_when_no_terms_match(self) -> None:
        self.assertEqual(search("태양계 행성", self.items), [])

    def test_search_rejects_empty_query_and_non_positive_top_k(self) -> None:
        with self.assertRaises(ValueError):
            search(" ", self.items)
        with self.assertRaises(ValueError):
            search("매출", self.items, top_k=0)


class SummarizerTests(unittest.TestCase):
    def test_summarize_items_preserves_original_metadata(self) -> None:
        create = Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  매출이 증가했다.  "))]
        ))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        item = {
            "id": "text-1",
            "source": "report.pdf",
            "page": 2,
            "kind": "text",
            "original": "매출은 전년 대비 증가했다.",
        }

        result = summarize_items([item], client=client)

        self.assertEqual(result[0]["summary"], "매출이 증가했다.")
        self.assertEqual(result[0]["original"], item["original"])
        self.assertEqual(result[0]["page"], 2)
        create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
