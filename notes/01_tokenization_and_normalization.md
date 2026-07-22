# LLM from Scratch & 한국어 LLM 연구 노트

이 노트는 **LLM 바닥부터 구현하기 (LLM from Scratch)** 과정을 학습하며 기록하는 연구 및 필기 노트입니다. 

---

## 📌 [특집] 한국어 토큰화 특화 분석: LG EXAONE Tokenizer

한국어는 영어와 언어적 구조가 달라 일반적인 LLM 토크나이저를 사용하면 효율성과 성능이 크게 떨어집니다. 최근 LG AI Research의 **EXAONE(엑사원)**이 한국어 토큰화에서 우수한 성능을 보이는 기술적 배경을 정리합니다.

### 1. 기존 LLM 토크나이저(Llama, GPT 등)의 한국어 문제점
*   **교착어(Agglutinative Language) 특성**: 한국어는 어근에 조사, 어미가 결합하는 구조입니다. (예: "공부하(어근) + 였다(어미) + 고(연결) + 는(조사)")
*   **바이트 폴백(Byte-fallback)의 비효율성**: Llama 등 영어 중심 모델의 토크나이저는 한국어 단어가 사전(Vocabulary)에 없으면 글자를 바이트(UTF-8 Byte) 단위로 쪼개서 처리합니다. 이로 인해 한 글자가 3개 이상의 토큰으로 분해되어 **토큰 수 급증 → 컨텍스트 윈도우 낭비 → 추론 비용 상승 및 속도 저하**로 이어집니다.

### 2. LG EXAONE 토크나이저가 잘하는 이유
*   **한국어 맞춤형 대용량 어휘 사전(Vocab)**: 한국어 코퍼스를 대규모로 분석하여 자주 쓰이는 형태소와 한국어 단어 조합을 사전에 미리 등록해 두었습니다. 바이트 폴백을 최소화하여 단어 형태 그대로 토큰화합니다.
*   **한국어/영어 균형 BPE**: 영어와 한국어 간의 토큰 압축률 균형을 맞춰, 다국어 처리 시 한국어 성능이 희생되지 않도록 설계되었습니다.
*   **글자 보존율(Token-to-Word Ratio)**: 동일한 한국어 문장을 토큰화했을 때 EXAONE은 타 모델 대비 훨씬 적은 수의 토큰을 생성합니다. (예: Llama에서 10토큰이 드는 문장을 EXAONE은 3~4토큰으로 압축 가능)

### 3. 토큰화 단위의 설계 방식 비교 (단어 vs 글자 vs 서브워드)
LLM 아키텍처 설계 시 토큰화 단위를 어떻게 정의하느냐에 따라 명확한 트레이드오프가 존재합니다.

*   **단어 단위 (Word-level)**:
    *   *문제점*: 사전(Vocabulary)의 크기가 무한정 커져 컴퓨터 메모리가 버티지 못하며, 사전에 없는 오타나 신조어가 나오면 모르는 단어인 `<unk>`로 처리하여 문맥을 이해하지 못합니다.
*   **글자 단위 (Character-level)**:
    *   *문제점*: 사전 크기는 자모음/알파벳 몇십 개 수준으로 압축되지만, 문장 하나를 표현할 때 시퀀스 길이가 너무 길어져 LLM의 컨텍스트 윈도우 한계를 금방 초과합니다.
*   **서브워드 단위 (Subword-level) — 현대 LLM 표준**:
    *   *해결책*: 자주 쓰이는 표현은 통째로 단어로 매핑하고, 드물게 쓰이는 표현은 쪼개서 매핑합니다. (예: `"킹사과"` ➡️ `[킹]` + `[사과]`). 사전 크기를 합리적으로 유지하면서도 OOV(Out-of-Vocabulary) 문제를 완벽히 해결합니다.

---

## ✍️ [Step 1] 텍스트 정규화 (Normalization) & 기초 토큰화

LLM 파이프라인의 가장 첫 단계로, 원시 텍스트를 깨끗하게 정형화하고 단어/문장부호 단위로 쪼개는 단계입니다.

### 1. 텍스트 정규화 (Normalization)
불필요한 공백, 탭, 과도한 줄바꿈을 제거하여 텍스트의 일관성을 확보합니다.

```python
import re

def normalize_text(text):
    # 앞뒤 공백 제거
    text = text.strip()
    
    # 탭(\t) 및 다중 공백을 단일 공백(' ')으로 변환
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 3개 이상 연속된 줄바꿈을 2개(\n\n)로 압축 (문단 구분 보존)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 각 행 시작 부분의 들여쓰기 공백 제거 (?m: 다중행 모드)
    text = re.sub(r'(?m)^[ ]+', '', text)
    
    return text
```

### 2. 문장부호 분리형 토큰화 (Basic Tokenization)
공백을 기준으로 쪼개되, 문장부호(`,`, `.`, `?`, `!` 등)가 단어 뒤에 붙어 오염되지 않도록 분리합니다.

```python
# 쪼갤 기준이 되는 문장부호들과 공백 패턴 정의
# 괄호 ()로 감싸야 re.split이 분리 기준이 된 문장부호도 버리지 않고 토큰 리스트에 포함시킵니다.
token_pattern = r'([,.:;?_!"()\']|--|\s)'

def tokenize(text):
    # 정규식 패턴 기준으로 분할
    raw_tokens = re.split(token_pattern, text)
    
    # 빈 토큰 및 공백만 있는 토큰 제외하고 strip 처리
    tokens = [t.strip() for t in raw_tokens if t.strip()]
    
    return tokens
```

---

## 📦 [Step 2] 파이토치(PyTorch) 데이터셋 및 데이터로더 구현

토큰화된 데이터를 바탕으로 GPT가 **"다음 토큰 예측(Next-token Prediction)"** 학습을 할 수 있도록 데이터를 슬라이딩 윈도우 방식으로 자르고, 파이토치 `Dataset`과 `DataLoader`로 가공하는 과정입니다.

### 1. 슬라이딩 윈도우와 타깃 데이터의 오프셋 시프트 (Shift)
GPT 학습 시 입력 데이터 $X$와 타깃 데이터 $Y$는 항상 **1 토큰만큼 오른쪽으로 이동(Shift)**해 있어야 합니다.
*   **입력 ($X$)**: `[In, the, beginning]` ➡️ `[ID_1, ID_2, ID_3]`
*   **타깃 ($Y$)**: `[the, beginning, ,]` ➡️ `[ID_2, ID_3, ID_4]` (입력의 다음 단어들)

#### ❓ 왜 +1 토큰 시프트가 중요한가? (이론적 의의)
1.  **자동회귀(Autoregressive) 학습의 본질**: GPT는 주어진 문맥 뒤에 올 "단 1개의 다음 토큰"을 맞추는 방식으로 세상을 배웁니다. 만약 시프트를 하지 않고 $X$와 $Y$를 동일하게 둔다면, 모델은 그냥 입력받은 단어를 그대로 복사하는 단순 복사(Copy) 기계가 되어 버립니다.
2.  **데이터 학습 효율성**: 문장을 통째로 넣고 타깃을 1칸 시프트해두면, 모델은 단 한 번의 연산(Forward pass)으로 문장 내 모든 단어 위치에서 각각 다음 단어를 맞추는 학습을 동시에(병렬로) 수행할 수 있습니다.
3.  **컨닝 방지 (Causal Masking)**: 입력 $X$에 뒷부분 정답 단어들이 미리 들어있어도, 모델 내부의 Causal Attention Mask가 미래 시점의 토큰들을 투명 장막으로 가려버리기 때문에, 모델은 앞단어만 보면서 정확히 +1 시프트된 타깃의 정답 단어를 안전하게 학습하게 됩니다.

### 2. 파이토치 커스텀 `Dataset` 구현
```python
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 1. 전체 텍스트 토큰화
        token_ids = tokenizer.encode(txt)

        # 2. 슬라이딩 윈도우 방식으로 데이터를 max_length 크기만큼 분할
        # i + max_length + 1 범위인 이유는 타깃 데이터가 1칸 뒤에 위치하기 때문입니다.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            
            # 파이토치 텐서로 변환하여 리스트에 축적
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # 데이터 샘플의 총 개수 반환
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 특정 인덱스의 입력과 타깃 한 쌍 반환
        return self.input_ids[idx], self.target_ids[idx]
```

### 3. 미니배치(Mini-batch) 구성을 위한 `DataLoader` 유틸리티
에폭(Epoch)마다 데이터를 섞고(Shuffle), 배치 크기만큼 묶어서 모델에 주입해 주는 데이터로더를 만듭니다.

```python
def create_dataloader_v1(txt, tokenizer, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 커스텀 데이터셋 인스턴스 생성
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    # 파이토치 데이터로더 생성
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, # 마지막 불완전한 배치를 버림 (학습 안정성)
        num_workers=num_workers # 데이터 로딩 병렬 프로세스 수
    )
    
    return dataloader
```

#### 💡 핵심 파라미터 이해
*   `max_length`: LLM이 한 번에 입력받아 연산할 **컨텍스트 윈도우 크기** (학습 데이터의 시퀀스 길이).
*   `stride`: 슬라이딩 윈도우가 한 번에 **이동할 단어 수**.
    *   `stride = 1`: 모든 위치를 꼼꼼히 학습하지만 연산 및 데이터셋 크기 증가.
    *   `stride = max_length`: 데이터셋 조각들 간에 겹치는 영역이 전혀 없음 (독립적).
*   `drop_last=True`: 남는 짜투리 데이터의 크기가 배치 크기보다 작을 때 이를 버려, 경사하강법 계산 시 배치 텐서 크기 불일치로 인한 오류를 방지합니다.

### 4. 데이터로더 출력 검증 (디코딩하여 눈으로 확인하기)
데이터로더가 뱉어내는 텐서(숫자 ID)를 다시 텍스트로 디코드해서 눈으로 정합성을 검증하는 코드입니다.

```python
# 데이터로더에서 배치 1개 꺼내기
data_iter = iter(dataloader)
x_batch, y_batch = next(data_iter)

# 배치 내의 첫 번째 샘플(인덱스 0) 추출
first_x = x_batch[0].tolist()
first_y = y_batch[0].tolist()

print("X (Input IDs)  :", first_x)
print("Y (Target IDs) :", first_y)

# 토크나이저를 사용해 다시 문자로 복원 (디코딩)
print("\nX (Input Text)  :", tokenizer.decode(first_x))
print("Y (Target Text) :", tokenizer.decode(first_y))
```

#### 🖥️ 실제 출력 예상 형태
1칸 시프트가 완벽히 적용되어, X의 두 번째 단어부터가 Y의 첫 번째 단어로 정합성이 맞물려 출력되는 것을 볼 수 있습니다.
```text
X (Input IDs)  : [59441, 102432, 2351]
Y (Target IDs) : [102432, 2351, 8749]

X (Input Text)  : In the beginning
Y (Target Text) : the beginning, there
```

---

## 🗺️ LLM from Scratch 학습 로드맵

앞으로 학습을 진행하며 이 노트에 아래 순서대로 개념을 추가하고 구현 코드를 누적해 나갑니다.

*   [x] **1. 텍스트 정규화 & 기초 토큰화** (완료)
*   [x] **2. 토큰 ID 매핑 & 데이터로더 구현** (완료)
*   [ ] **3. 어텐션 매커니즘 (Attention)**
    *   Self-Attention 구조 이해
    *   Causal Attention (Masked Attention) 구현
    *   Multi-Head Attention 클래스 설계
*   [ ] **4. Transformer 블록 및 GPT 아키텍처**
    *   Layer Normalization & GELU 활성화 함수
    *   Shortcut Connection (잔차 연결) 구현
    *   전체 GPT 모델 클래스 어셈블리
*   [ ] **5. 프리트레이닝 (Pre-training) & 손실 함수**
    *   Cross Entropy Loss 계산
    *   학습 루프 설계
*   [ ] **6. 파인튜닝 (SFT & DPO)**
    *   지시어 학습(Instruction tuning) 및 선호도 최적화 원리

