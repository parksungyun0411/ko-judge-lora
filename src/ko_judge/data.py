"""KUDGE 내려받기·불러오기.

원본은 저장소에 넣지 않고 고정한 커밋에서 내려받아 sha256으로 확인한다.
- 평가 4종: GitHub guijinSON/kudge (LICENSE: MIT) 커밋 09e1316의 data/k2-eval-*.csv
  HF HAERAE-HUB/KUDGE 리비전 7168dc8의 kudge-*.csv와 바이트 단위로 같다(sha256 동일).
- 원 주석(raw): HF에만 있다. 데이터셋 카드에 라이선스 표기가 없어 프로파일링에만 쓴다.

  python -m ko_judge.data
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GITHUB_COMMIT = "09e13166f9f6a227cabaf82b177342a53c507d40"
HF_REVISION = "7168dc844bb043fe0b0b0ee860da4dd20a67e5a2"
GITHUB_RAW = f"https://raw.githubusercontent.com/guijinSON/kudge/{GITHUB_COMMIT}"
HF_RAW = f"https://huggingface.co/datasets/HAERAE-HUB/KUDGE/resolve/{HF_REVISION}"
DATA_DIR = Path("data/kudge")


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str


SOURCES = {
    "pointwise": Source(
        "pointwise.csv",
        f"{GITHUB_RAW}/data/k2-eval-pointwise.csv",
        "88c534e040ba7d43e2e668f220e4c90a3f204f6fc68582804e6cd972169774d9",
    ),
    "pairwise": Source(
        "pairwise.csv",
        f"{GITHUB_RAW}/data/k2-eval-pairwise.csv",
        "27f79ae76a383e5c0f5eea6f9798cbddb0b6c38da7d99bc4c59f5f96dcd003a0",
    ),
    "pointwise_false": Source(
        "pointwise_falseinfo.csv",
        f"{GITHUB_RAW}/data/k2-eval-pointwise-falseinfo.csv",
        "6443a47040c53f9b29010522a85b8a9a120b8d197b7bb35bd702d5d32a5438ae",
    ),
    "pairwise_false": Source(
        "pairwise_falseinfo.csv",
        f"{GITHUB_RAW}/data/k2-eval-pairwise-falseinfo.csv",
        "66abe792a769a692289071e9da031ee64ac3bdd9b9be2b5b1c258c9f674727d0",
    ),
    "human_raw": Source(
        "human_annotation_raw.csv",
        f"{HF_RAW}/kudge-human-annotation-raw.csv",
        "f3c30c9429d8321bf784e57f791a4a4b5373be361bb40a90d0c7e69ae28d045e",
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path.name} sha256 불일치: {actual[:16]} != {expected[:16]}")


def download(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, src in SOURCES.items():
        path = data_dir / src.name
        if not path.exists():
            urllib.request.urlretrieve(src.url, path)
        check_hash(path, src.sha256)
        out[key] = path
    return out


def load(key: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / SOURCES[key].name
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — `make data`로 먼저 내려받는다")
    df = pd.read_csv(path)
    if key in {"pointwise_false", "pairwise_false"}:
        col = "type" if "type" in df.columns else "position"
        df[col] = df[col].str.replace("\x08", "", regex=False)  # 원본에 백스페이스 문자 2건
    return df


def load_pointwise(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """행 번호를 row_id로 붙인다. 이후 분할·예측은 모두 row_id로 잇는다."""
    df = load("pointwise", data_dir)
    df.insert(0, "row_id", range(len(df)))
    return df


def main() -> None:
    paths = download()
    for k, p in paths.items():
        print(f"{k:16s} {p}  sha256={sha256_file(p)[:16]}")


if __name__ == "__main__":
    main()
