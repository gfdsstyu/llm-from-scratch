# llm-from-scratch

LLM을 밑바닥부터(from scratch) 구현하며 정리하는 개인 학습 레포.
토큰화·정규화부터 시작해 어텐션·트랜스포머·학습 루프까지 단계별로 쌓아갑니다.

## 구성

```
llm-from-scratch/
├── notes/     # 단계별 학습 노트 (Markdown, 파이썬 코드 인라인)
│   └── 01_tokenization_and_normalization.md
└── src/       # 실행 가능한 파이썬 구현 (노트에서 뽑아낸 코드)
```

## 노트 목록

| # | 주제 | 파일 |
|---|------|------|
| 01 | 텍스트 정규화 · 기초 토큰화 · 한국어 토크나이저(EXAONE) | [notes/01_tokenization_and_normalization.md](notes/01_tokenization_and_normalization.md) |

## 참고

- 한국어 토큰화 특화(교착어·바이트 폴백·서브워드 BPE) 분석 포함
- 코드 예제는 파이썬 3.x 기준
