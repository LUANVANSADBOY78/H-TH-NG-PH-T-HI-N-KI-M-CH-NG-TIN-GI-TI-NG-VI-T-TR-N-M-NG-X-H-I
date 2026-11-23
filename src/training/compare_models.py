"""
Module So sánh Đánh giá Đa Mô hình (Model Benchmarking).
Đọc kết quả thực nghiệm từ hệ thống xuất ra, loại bỏ mọi số hard-code.
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_comparison import get_comparison_summary

def run_model_comparison() -> pd.DataFrame:
    return get_comparison_summary()

if __name__ == "__main__":
    print(run_model_comparison().to_string(index=False))
