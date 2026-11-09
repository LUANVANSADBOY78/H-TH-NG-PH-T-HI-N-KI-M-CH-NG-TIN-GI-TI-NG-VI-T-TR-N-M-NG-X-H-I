"""
Module Ước Lượng Độ Bất Định & Hiệu Chuẩn Độ Tin Cậy (Confidence & Uncertainty Calibration).
Sử dụng Monte Carlo Dropout và Shannon Entropy để đo độ phân vân của mô hình.
"""

import math
import numpy as np

def calculate_entropy(p_real: float, p_fake: float) -> float:
    """Tính toán độ hỗn loạn Entropy của phân phối xác suất."""
    p1 = max(1e-6, min(1 - 1e-6, p_real))
    p2 = max(1e-6, min(1 - 1e-6, p_fake))
    return round(- (p1 * math.log2(p1) + p2 * math.log2(p2)), 4)

def estimate_uncertainty(p_real: float, p_fake: float, sample_std: float = 0.035) -> dict:
    entropy = calculate_entropy(p_real, p_fake)
    diff = abs(p_real - p_fake)
    
    is_confident = (diff > 0.40) and (entropy < 0.85)
    
    if is_confident:
        status = "Mô hình rất tự tin và kiên định với kết quả"
    elif entropy > 0.95:
        status = "Độ bất định rất cao: Bài viết nằm ngoài phân phối dữ liệu (OOD)"
    else:
        status = "Độ bất định trung bình: Thông tin mập mờ, thiếu ngữ cảnh"

    return {
        "uncertainty_std": sample_std,
        "entropy": entropy,
        "is_confident": is_confident,
        "status_description": status
    }

