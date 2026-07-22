"""
Step 1 — 텍스트 정규화(Normalization) & 기초 토큰화(Basic Tokenization).

LLM 파이프라인의 첫 단계. 원시 텍스트를 일관된 형태로 정규화하고,
공백/문장부호 단위로 쪼개 토큰 리스트를 만든다.

노트: notes/01_tokenization_and_normalization.md
의존성: 없음 (표준 라이브러리 re 만 사용)

실행:
    python src/step01_tokenization.py
"""
import re


def normalize_text(text: str) -> str:
    """불필요한 공백·탭·과도한 줄바꿈을 정리해 텍스트 일관성을 확보한다.

    Args:
        text: 원시 입력 문자열.

    Returns:
        정규화된 문자열. 문단 구분(빈 줄)은 보존한다.
    """
    # 앞뒤 공백 제거
    text = text.strip()
    # 탭 및 다중 공백 → 단일 공백
    text = re.sub(r'[ \t]+', ' ', text)
    # 3개 이상 연속 줄바꿈 → 2개로 압축 (문단 구분 보존)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 각 행 시작부의 들여쓰기 제거 (?m: 다중행 모드)
    text = re.sub(r'(?m)^[ ]+', '', text)
    return text


# 분할 기준이 되는 문장부호/공백 패턴.
# 캡처 그룹 ()로 감싸야 re.split이 분리 기준(문장부호)도 버리지 않고 토큰에 포함시킨다.
TOKEN_PATTERN = r'([,.:;?_!"()\']|--|\s)'


def tokenize(text: str) -> list:
    """정규식 기준으로 텍스트를 토큰 리스트로 분할한다.

    문장부호가 단어 뒤에 붙어 오염되지 않도록 분리한다.

    Args:
        text: 토큰화할 문자열.

    Returns:
        공백을 제외한 토큰 문자열 리스트.
    """
    raw_tokens = re.split(TOKEN_PATTERN, text)
    # 빈 토큰 및 공백만 있는 토큰 제외
    return [t.strip() for t in raw_tokens if t.strip()]


if __name__ == "__main__":
    sample = "   Hello,   world!  This  is\t\tan   example.\n\n\n\nLLM from scratch.  "
    print("원본     :", repr(sample))

    normalized = normalize_text(sample)
    print("정규화   :", repr(normalized))

    tokens = tokenize(normalized)
    print("토큰     :", tokens)
    print("토큰 수  :", len(tokens))
