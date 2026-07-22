# llm-from-scratch

LLM을 밑바닥부터(from scratch) 구현하며 정리하는 개인 학습 레포.
토큰화·정규화부터 시작해 어텐션·트랜스포머·학습 루프까지 단계별로 쌓아갑니다.

## 구성

```
llm-from-scratch/
├── notes/            # 단계별 학습 노트 (Markdown, 이론 + 코드 설명)
│   └── 01_tokenization_and_normalization.md
├── src/              # 노트에서 뽑아낸 실행 가능한 파이썬 구현
│   ├── step01_tokenization.py   # 정규화 + 기초 토큰화 (의존성 없음)
│   └── step02_dataloader.py     # PyTorch Dataset/DataLoader (다음 토큰 예측)
└── requirements.txt  # torch, tiktoken (Step 2 이상)
```

## 실행

```bash
# Step 1 — 표준 라이브러리만으로 즉시 실행
python src/step01_tokenization.py

# Step 2 — 의존성 설치 후 실행
pip install -r requirements.txt
python src/step02_dataloader.py
```

## 노트 목록

| # | 주제 | 노트 | 코드 |
|---|------|------|------|
| 01 | 텍스트 정규화 · 기초 토큰화 · 한국어 토크나이저(EXAONE) | [notes/01](notes/01_tokenization_and_normalization.md) | [step01](src/step01_tokenization.py) |
| 02 | 토큰 ID 매핑 · 슬라이딩 윈도우 · DataLoader | [notes/01](notes/01_tokenization_and_normalization.md) | [step02](src/step02_dataloader.py) |

## 참고

- 한국어 토큰화 특화(교착어·바이트 폴백·서브워드 BPE) 분석 포함
- 코드 예제는 파이썬 3.x 기준
