# src/loader.py
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT as CONFIG_PROJECT_ROOT
from src.config import RAW_DATA_DIR

PROJECT_ROOT = CONFIG_PROJECT_ROOT
RAW_DIR = RAW_DATA_DIR


def find_raw_file(filename: str, raw_dir: Path = RAW_DIR) -> Path:
    """Tìm file trong data/raw (kể cả trong thư mục con, vd: expedia-hotel-rec...)."""
    matches = list(raw_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Không tìm thấy '{filename}' trong {raw_dir}")
    return matches[0]


def load_csv(
    path: str | Path,
    nrows: int | None = None,
    usecols: Sequence[str] | None = None,
    parse_dates: Iterable[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Hàm đọc CSV dùng chung, có kiểm tra đường dẫn."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        nrows=nrows,
        usecols=usecols,
        parse_dates=list(parse_dates) if parse_dates else None,
        **kwargs,
    )


DATASETS: dict[str, dict] = {
    "expedia_train": {
        "filename": "train.csv",
        "subdir": "expedia-hotel-recommendations",
        "parse_dates": ["date_time", "srch_ci", "srch_co"],
    },
    "expedia_test": {
        "filename": "test.csv",
        "subdir": "expedia-hotel-recommendations",
        "parse_dates": ["date_time", "srch_ci", "srch_co"],
    },
    "expedia_destinations": {
        "filename": "destinations.csv",
        "subdir": "expedia-hotel-recommendations",
    },
    "stander": {
        "filename": "train_ver2.csv",
        "subdir": "santander-product-recommendation",
    },
}


def load_dataset(name: str, nrows: int | None = None, **kwargs) -> pd.DataFrame:
    """Tải dataset theo tên đã đăng ký trong DATASETS."""
    if name not in DATASETS:
        raise KeyError(f"Dataset '{name}' không tồn tại. Có sẵn: {list(DATASETS)}")
    cfg = DATASETS[name].copy()
    subdir = cfg.pop("subdir", None)
    filename = cfg.pop("filename")
    base = RAW_DIR / subdir if subdir else RAW_DIR
    path = find_raw_file(filename, base)
    cfg.update(kwargs)  # tham số truyền vào ghi đè cấu hình mặc định
    return load_csv(path, nrows=nrows, **cfg)


def load_expedia_train(**kwargs) -> pd.DataFrame:
    """Lối tắt: tải train.csv của Expedia."""
    return load_dataset("expedia_train", **kwargs)
