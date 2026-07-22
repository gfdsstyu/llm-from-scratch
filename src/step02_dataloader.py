"""
Step 2 — PyTorch Dataset / DataLoader 구현 (다음 토큰 예측).

토큰 ID 시퀀스를 슬라이딩 윈도우로 잘라, 입력 X와 +1 시프트된 타깃 Y 쌍을
만드는 GPT 스타일 데이터 파이프라인.

  입력 X : [ID_1, ID_2, ID_3]
  타깃 Y : [ID_2, ID_3, ID_4]   (X 를 1칸 오른쪽으로 시프트)

노트: notes/01_tokenization_and_normalization.md
의존성: torch, tiktoken  (pip install -r requirements.txt)

실행:
    python src/step02_dataloader.py
"""
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """슬라이딩 윈도우로 (입력, +1 시프트 타깃) 쌍을 생성하는 데이터셋."""

    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Args:
            txt: 원시 텍스트.
            tokenizer: .encode(str) -> list[int] 를 제공하는 토크나이저(예: tiktoken).
            max_length: 컨텍스트 윈도우 크기(시퀀스 길이).
            stride: 슬라이딩 윈도우가 한 번에 이동할 토큰 수.
        """
        self.input_ids = []
        self.target_ids = []

        # 1. 전체 텍스트 토큰화
        token_ids = tokenizer.encode(txt)

        # 2. 슬라이딩 윈도우로 max_length 크기만큼 분할.
        #    타깃이 1칸 뒤라서 i + max_length + 1 까지 참조한다.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """데이터 샘플의 총 개수."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """특정 인덱스의 (입력, 타깃) 한 쌍을 반환한다."""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, tokenizer, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """GPTDatasetV1을 감싼 PyTorch DataLoader를 생성한다.

    Args:
        batch_size: 미니배치 크기.
        max_length: 컨텍스트 윈도우 크기.
        stride: 윈도우 이동 간격. max_length와 같으면 조각 간 겹침이 없다.
        shuffle: 에폭마다 데이터 섞기 여부.
        drop_last: 마지막 불완전 배치 폐기(텐서 크기 불일치로 인한 오류 방지).
        num_workers: 데이터 로딩 병렬 프로세스 수.

    Returns:
        torch.utils.data.DataLoader 인스턴스.
    """
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    try:
        import tiktoken
    except ImportError:
        raise SystemExit("tiktoken 이 필요합니다:  pip install -r requirements.txt")

    # 책과 동일하게 GPT-2 BPE 토크나이저 사용
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "In the beginning there was code, and the code became a language model."

    dataloader = create_dataloader_v1(
        text, tokenizer, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    # 배치 1개 꺼내 +1 시프트 정합성 눈으로 검증
    data_iter = iter(dataloader)
    x_batch, y_batch = next(data_iter)
    first_x = x_batch[0].tolist()
    first_y = y_batch[0].tolist()

    print("X (Input IDs)  :", first_x)
    print("Y (Target IDs) :", first_y)
    print("\nX (Input Text)  :", tokenizer.decode(first_x))
    print("Y (Target Text) :", tokenizer.decode(first_y))
