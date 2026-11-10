# 멀티모달 RAG — 논문 PDF의 표·그림까지 근거로 답하기

## 1. 문제 정의

- 분석 자료: Yang Song, Stefano Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution* (NeurIPS 2019, [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)), 23쪽
- 이 논문의 핵심 결과는 **본문이 아니라 표와 그림에 있습니다.**
  - CIFAR-10 정량 비교(Inception, FID)는 **Table 1**에만 모델별로 정리돼 있습니다.
  - 방법의 동기(데이터가 적은 저밀도 영역에서 score 추정이 부정확함)는 **Figure 1·2**, annealed Langevin dynamics의 효과는 **Figure 3**, 생성·인페인팅 결과는 **Figure 4~6**과 부록 그림으로 제시됩니다.
- 텍스트만 추출하는 RAG는 표가 한 줄 문자열로 깨지고 그림은 통째로 빠집니다. 그래서 "Table 1에서 NCSN보다 점수가 높은 모델은?", "Figure 2의 빨간 사각형은 무엇인가?" 같은 질문에 근거를 찾지 못합니다.
- 목표: 텍스트·표·그림을 모두 검색 대상으로 만들고, 답변할 때는 **원본 표·그림을 LLM이 직접 보게** 해서 근거 있는 한국어 답변을 만듭니다.

## 2. 아키텍처

```mermaid
flowchart TB
    PDF["data/invest.pdf"]

    subgraph S1["1단계 · 추출 (src/extract.py)"]
        P["Unstructured partition_pdf<br/>hi_res · by_title"]
        T["텍스트 블록"]
        TB["표 (HTML)"]
        IMG["그림 jpg → data/figures/"]
        P --> T & TB & IMG
    end

    subgraph S2["2단계 · 요약 (src/summarize.py)"]
        ST["텍스트·표 요약<br/>prompt | gpt-4o-mini | parser"]
        SI["그림 요약<br/>base64 이미지 → gpt-4o-mini 비전"]
    end

    subgraph S3["3단계 · 검색과 생성 (src/retriever.py + 노트북 Part 5·6)"]
        Q["한국어 질문"] --> TR1["번역 체인 → 영어"]
        TR1 --> VS[("Chroma<br/>요약 임베딩")]
        VS -- "doc_id" --> DS[("InMemoryStore<br/>원본: 텍스트·표 HTML·그림 base64")]
        DS --> LLM["RAG 체인<br/>gpt-4o-mini (텍스트 + 표 + 그림 입력)"]
        LLM --> TR2["번역 체인 → 한국어"]
        TR2 --> A["한국어 답변 + 근거 페이지"]
    end

    PDF --> P
    T & TB --> ST
    IMG --> SI
    ST & SI --> VS
    T & TB & IMG --> DS
```

### 왜 요약으로 검색하고 원본으로 생성하나요?

- **검색을 요약으로 하는 이유**: 표 HTML은 태그와 숫자가 대부분이라 그대로 임베딩하면 "무엇을 비교한 표인지"가 잘 드러나지 않습니다. 그림은 텍스트 임베딩을 아예 할 수 없습니다. 요약은 표·그림의 의미를 문장으로 바꿔 주기 때문에 질문과 같은 공간에서 비교할 수 있습니다.
- **생성을 원본으로 하는 이유**: 요약은 짧게 줄이는 과정에서 수치(예: `8.87 ± .12`)나 그림의 세부가 빠질 수 있습니다. 검색은 요약으로 하되, 답변 단계에서는 `doc_id`로 원본을 꺼내 LLM이 표 전체와 그림 자체를 보게 합니다.
- 두 저장소는 `MultiVectorRetriever`가 `doc_id` 하나로 연결합니다.

## 3. 기술 선택 근거

| 항목 | 선택 | 근거 |
| --- | --- | --- |
| 파서 | Unstructured `partition_pdf`, `strategy="hi_res"` | `fast`는 텍스트만 추출합니다. 표 구조(`infer_table_structure`)와 그림 추출(`extract_images_in_pdf`)은 레이아웃 모델을 쓰는 `hi_res`에서만 됩니다. |
| 청크 | `by_title`, 4000 / 3800 / 2000자 | 논문은 섹션 제목 단위로 내용이 나뉘어 있어 제목 기준이 의미를 보존합니다. 블록이 요약 단위가 되므로 섹션 하나가 통째로 들어가도록 크게 잡았습니다 (수업 01 설정). |
| 요약·생성 모델 | `gpt-4o-mini` | 텍스트와 이미지 입력을 모두 지원해 요약·비전·답변·번역을 한 모델로 처리하고, 블록·그림 수십 개를 요약해도 비용이 낮습니다. |
| 임베딩 | `text-embedding-3-small` | 텍스트 요약을 검색하므로 텍스트 임베딩으로 충분합니다. |
| 그림 검색 방식 | CLIP 대신 **그림 요약 + 텍스트 임베딩** | FashionRAG처럼 CLIP으로 그림을 직접 검색할 수도 있지만, CLIP은 일반 사진–캡션으로 학습돼 그래프·샘플 격자처럼 논문 그림의 의미("저밀도 영역에서 score가 부정확")를 잘 잡지 못합니다. 그림을 문장으로 요약하면 텍스트·표와 같은 검색기 하나로 다룰 수 있습니다. |
| 프롬프트 언어 | 요약·답변 프롬프트는 **영어**, 질문·답변은 **번역 체인**으로 한↔영 | 논문 원문이 영어라 영어 요약이 원문 용어(score matching, Langevin dynamics 등)를 그대로 유지합니다. 한국어 질문은 FashionRAG처럼 번역 체인으로 영어로 바꿔 검색하고, 답변은 다시 한국어로 번역합니다. |
| 벡터 DB | Chroma (메모리) | 문서 한 편 규모라 별도 서버 없이 충분합니다. 추출·요약 결과는 pickle로 저장해 재실행 비용을 줄입니다. |

## 4. 실행 방법

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env   # .env에 OPENAI_API_KEY 입력
```

1. Poppler와 Tesseract를 설치하고 PATH에 추가합니다 (`hi_res` 파싱에 필요).
2. 분석 PDF는 `data/invest.pdf`로 저장소에 포함돼 있습니다 (출처: 위 arXiv 링크).
3. `notebooks/Multi_modal_RAG.ipynb`를 위에서부터 실행합니다.

```text
multimodal-rag/
├── .venv/                      # 가상환경 (Git 제외)
├── .env                        # API 키 (Git 제외, 절대 커밋 금지)
├── .env.example                # 키 없는 템플릿 (Git 포함)
├── .gitignore
├── requirements.txt            # 의존성 목록
├── requirements.lock.txt       # 버전 고정 파일
├── README.md
├── data/
│   ├── invest.pdf              # 원본 PDF
│   └── figures/                # 추출된 이미지 (Git 제외, 용량 큼)
├── notebooks/
│   └── Multi_modal_RAG.ipynb   # 실습 노트북 (번역 체인·RAG 체인·전체 파이프라인 포함)
└── src/
    ├── extract.py              # PDF 파싱 함수
    ├── summarize.py            # 요약 체인
    └── retriever.py            # 다중 벡터 검색기
```

노트북을 실행하면 추출·요약 결과가 `data/processed/`에 pickle로 저장됩니다 (Git 제외).

## 5. 질의응답 5건

<!-- 노트북 마지막 셀이 출력한 표를 그대로 붙여 넣으세요. -->

| # | 질문 | 답변 | 근거가 된 원본 종류 |
| --- | --- | --- | --- |
| 1 | CIFAR-10 unconditional 설정에서 NCSN의 Inception score와 FID는 각각 얼마인가요? | (실행 후 작성) | |
| 2 | Table 1에서 Inception score가 NCSN보다 높은 모델은 무엇이고, 어떤 설정인가요? | (실행 후 작성) | |
| 3 | Figure 2에서 빨간 사각형으로 표시된 영역은 무엇을 의미하나요? | (실행 후 작성) | |
| 4 | Figure 3에서 annealed Langevin dynamics와 일반 Langevin dynamics의 결과는 어떻게 다른가요? | (실행 후 작성) | |
| 5 | 이미지 생성 실험에서 노이즈 수준 σ는 몇 개, 어떤 범위로 설정했나요? | (실행 후 작성) | |

## 6. 한계

- 수식이 많은 논문이라 텍스트 추출 시 위첨자·아래첨자·그리스 문자가 깨진 채로 들어갑니다 (예: `qσ2 (x)`). 수식 관련 질문은 정확도가 떨어질 수 있습니다.
- 논문 옆 세로 arXiv 표기 같은 텍스트가 글자 단위로 흩어져 첫 블록에 노이즈로 섞입니다.
- `hi_res` 레이아웃 인식 결과에 따라 그림 하나가 여러 조각으로 잘리거나, 캡션이 그림과 다른 블록으로 분리될 수 있습니다.
- 그림 요약은 비전 모델의 해석이라 작은 글자·범례를 잘못 읽을 수 있습니다. 답변 단계에서 원본 그림을 다시 보여 주는 것으로 일부 보완합니다.
- 번역을 두 번 거치므로 질문의 뉘앙스나 용어가 바뀔 수 있고, 질문 하나에 API 호출이 3번 발생합니다.
- 상위 k=4개만 검색하므로, 여러 표·그림을 함께 비교해야 하는 질문은 근거가 일부 빠질 수 있습니다.

## 7. 보안

- API 키는 `.env`에만 두고 `.gitignore`로 제외합니다. 저장소에는 키 이름만 있는 `.env.example`만 포함합니다.
- 노트북에는 키 앞 4자리만 출력합니다.
