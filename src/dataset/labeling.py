"""
Module Quản lý & Ánh xạ Nhãn Dữ liệu (Labeling).
"""

LABEL_MAP = {
    0: "Supported",
    1: "Refuted",
    2: "NEI"
}

REVERSE_LABEL_MAP = {
    "supported": 0,
    "thật": 0,
    "true": 0,
    "real": 0,
    "refuted": 1,
    "giả": 1,
    "fake": 1,
    "nei": 2,
    "chưa rõ": 2,
    "not enough info": 2
}

def standardize_label(val) -> int:
    """Quy chuẩn nhãn bất kỳ về mã số 0 (Supported), 1 (Refuted), 2 (NEI)."""
    if isinstance(val, (int, float)):
        return int(val)
    str_val = str(val).strip().lower()
    return REVERSE_LABEL_MAP.get(str_val, 1)

def get_label_name(label_id: int) -> str:
    return LABEL_MAP.get(label_id, "Unknown")

