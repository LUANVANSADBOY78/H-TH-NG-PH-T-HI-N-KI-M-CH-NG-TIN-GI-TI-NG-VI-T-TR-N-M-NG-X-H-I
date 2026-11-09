"""
Module Phân Tích Xu Hướng & Cảnh Báo Bùng Phát Tin Giả (Trend Analysis & Early Warning System).
Đáp ứng các yêu cầu:
1. Phân tích chuỗi thời gian (Time-series analysis) & Phát hiện đợt bùng phát tin giả bất thường (Spike Detection).
2. Thống kê Hashtag phổ biến và Tần suất từ khóa nhạy cảm.
3. Cảnh báo theo tần suất và chủ đề (Y tế, Tài chính, Chính trị, Việc làm lừa đảo).
4. Cảnh báo theo tài khoản nghi vấn (User Flagging) và Khu vực địa lý.
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 1. TẠO DỮ LIỆU MẪU CHUỖI THỜI GIAN THEO DÕI TIN GIẢ (30 NGÀY GẦN NHẤT)
def get_simulated_time_series_data() -> pd.DataFrame:
    """Tạo hoặc đọc dữ liệu chuỗi thời gian 30 ngày ghi nhận tin tức MXH."""
    np.random.seed(42)
    end_date = datetime.now()
    dates = [(end_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(29, -1, -1)]
    
    # Số lượng tin thật dao động ổn định
    real_counts = np.random.normal(loc=120, scale=15, size=30).astype(int)
    # Số lượng tin giả nền tảng thấp
    fake_counts = np.random.normal(loc=25, scale=8, size=30).astype(int)
    fake_counts = np.clip(fake_counts, 5, None)
    
    # Giả lập 2 đợt bùng phát tin giả (Spikes)
    # Spike 1: Ngày thứ 12 (ví dụ tin giả y tế / bệnh dịch)
    fake_counts[12] = 115
    fake_counts[13] = 95
    # Spike 2: Ngày thứ 25 (ví dụ lừa đảo đầu tư / bão lũ)
    fake_counts[25] = 140
    fake_counts[26] = 120

    df = pd.DataFrame({
        "date": dates,
        "real_news": real_counts,
        "fake_news": fake_counts,
        "total": real_counts + fake_counts
    })
    
    # Tính Z-Score cho fake_news để làm thuật toán Spike Detection
    mean_fake = df["fake_news"].mean()
    std_fake = df["fake_news"].std()
    df["z_score"] = (df["fake_news"] - mean_fake) / (std_fake if std_fake > 0 else 1)
    df["is_spike"] = df["z_score"] > 2.0  # Ngưỡng lệch chuẩn > 2.0
    
    return df

def generate_time_series_chart() -> str:
    """Vẽ biểu đồ chuỗi thời gian và đánh dấu đợt tăng bất thường (Spike Detection)."""
    df = get_simulated_time_series_data()
    out_path = os.path.join(OUTPUTS_DIR, "trend_time_series.png")
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    
    x = range(len(df))
    ax.plot(x, df["real_news"], label="Tin Thật / Kiểm chứng (Reliable)", color="#28a745", linewidth=2.2, marker="o", markersize=4)
    ax.plot(x, df["fake_news"], label="Tin Giả / Độc hại (Unreliable)", color="#dc3545", linewidth=2.8, marker="s", markersize=5)
    
    # Ngưỡng cảnh báo trung bình + 2*sigma
    mean_val = df["fake_news"].mean()
    std_val = df["fake_news"].std()
    threshold = mean_val + 2 * std_val
    ax.axhline(threshold, color="#ff9800", linestyle="--", linewidth=1.5, label=f"Ngưỡng Cảnh Báo Spike (Threshold = {threshold:.1f})")
    
    # Đánh dấu các điểm Spike bùng phát bất thường
    spike_indices = df.index[df["is_spike"]].tolist()
    for idx in spike_indices:
        val = df.loc[idx, "fake_news"]
        date_str = df.loc[idx, "date"]
        ax.scatter([idx], [val], color="#d63031", s=180, zorder=5, edgecolors="black", linewidth=1.5)
        ax.annotate(
            f"[BÙNG PHÁT ĐỘT BIẾN!]\n({date_str}: {val} ca)",
            xy=(idx, val),
            xytext=(idx - 2, val + 15),
            arrowprops=dict(facecolor="#d63031", shrink=0.08, width=1.5, headwidth=6),
            fontweight="bold",
            color="#d63031",
            fontsize=8.5
        )

    ax.set_title("XU HƯỚNG LAN TRUYỀN TIN TỨC & PHÁT HIỆN BÙNG PHÁT BẤT THƯỜNG (SPIKE DETECTION)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Thời gian theo ngày (30 ngày gần nhất)", fontsize=9.5)
    ax.set_ylabel("Số lượng bài đăng ghi nhận", fontsize=9.5)
    
    step = 3
    ax.set_xticks(list(x)[::step])
    ax.set_xticklabels([df["date"].iloc[i][5:] for i in list(x)[::step]], rotation=35, fontsize=8.5)
    
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def generate_hashtag_keyword_chart() -> str:
    """Vẽ biểu đồ cột thống kê Top 10 Hashtag & Từ khóa nhạy cảm có tần suất xuất hiện cao nhất."""
    out_path = os.path.join(OUTPUTS_DIR, "trend_keywords.png")
    
    keywords_data = [
        {"item": "#thần_dược_ung_thư", "count": 420, "topic": "Y tế & Dược phẩm", "risk": "Cao"},
        {"item": "việc_nhẹ_lương_cao", "count": 385, "topic": "Lừa đảo việc làm", "risk": "Cao"},
        {"item": "#kiếm_tiền_online_app", "count": 340, "topic": "Tài chính đa cấp", "risk": "Cao"},
        {"item": "chữa_dứt_điểm_100%", "count": 290, "topic": "Y tế & Thực phẩm", "risk": "Cao"},
        {"item": "#bà_con_chú_ý_gấp", "count": 275, "topic": "Giật gân kích động", "risk": "Trung bình"},
        {"item": "cắt_lỗ_đất_nền", "count": 230, "topic": "Bất động sản ảo", "risk": "Trung bình"},
        {"item": "#rơi_máy_bay_thảm_họa", "count": 210, "topic": "Tin đồn tai nạn", "risk": "Cao"},
        {"item": "xóa_bài_ngay_lập_tức", "count": 185, "topic": "Thao túng tâm lý", "risk": "Trung bình"},
        {"item": "#nạp_tiền_nhận_hoa_hồng", "count": 170, "topic": "Lừa đảo nạp tiền", "risk": "Cao"},
        {"item": "kịch_độc_chết_người", "count": 155, "topic": "Cảnh báo giả mạo", "risk": "Trung bình"}
    ]
    
    df_kw = pd.DataFrame(keywords_data)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    
    colors = ["#d63031" if r == "Cao" else "#e17055" for r in df_kw["risk"]]
    bars = ax.barh(df_kw["item"][::-1], df_kw["count"][::-1], color=colors[::-1], height=0.65)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 6, bar.get_y() + bar.get_height()/2, f"{int(w)} lượt", va="center", ha="left", fontsize=8.5, fontweight="bold", color="#2d3436")
        
    ax.set_title("TOP 10 HASHTAG & TỪ KHÓA NHẠY CẢM XUẤT HIỆN TẦN SUẤT CAO NHẤT", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Số lượt xuất hiện trong các bài đăng bị gắn nhãn tin giả", fontsize=9.5)
    ax.set_xlim(0, max(df_kw["count"]) * 1.18)
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def get_topic_alert_summary() -> pd.DataFrame:
    """Tạo bảng phân tích mức độ cảnh báo theo chủ đề nhạy cảm."""
    data = [
        {
            "Chủ đề theo dõi": "🏥 Y tế, Thuốc & Thực phẩm chức năng",
            "Số ca tin giả": 710,
            "Tỷ lệ tăng tuần": "+42.5%",
            "Mức độ cảnh báo": "🔴 BÁO ĐỘNG ĐỎ (Bùng phát thần dược giả)",
            "Hành động đề xuất": "Chuyển dữ liệu sang Bộ Y Tế, ưu tiên thẩm định đơn thuốc"
        },
        {
            "Chủ đề theo dõi": "💼 Tuyển dụng & Lừa đảo việc nhẹ lương cao",
            "Số ca tin giả": 590,
            "Tỷ lệ tăng tuần": "+28.1%",
            "Mức độ cảnh báo": "🔴 BÁO ĐỘNG ĐỎ (Mạo danh doanh nghiệp KCN)",
            "Hành động đề xuất": "Quét số tài khoản ngân hàng và hợp đồng cọc tiền"
        },
        {
            "Chủ đề theo dõi": "💰 Đầu tư tài chính, Đa cấp & App ảo",
            "Số ca tin giả": 450,
            "Tỷ lệ tăng tuần": "+15.3%",
            "Mức độ cảnh báo": "🟡 CẢNH BÁO VÀNG (Cần theo dõi sát)",
            "Hành động đề xuất": "Chặn link miền độc hại, kiểm tra giấy phép sàn"
        },
        {
            "Chủ đề theo dõi": "🏠 Bất động sản, Thuê phòng trọ ảo",
            "Số ca tin giả": 210,
            "Tỷ lệ tăng tuần": "-4.2%",
            "Mức độ cảnh báo": "🟢 BÌNH THƯỜNG (Ổn định)",
            "Hành động đề xuất": "Khuyến cáo người thuê chỉ giao dịch trực tiếp"
        },
        {
            "Chủ đề theo dõi": "🎓 Giáo dục, Cuộc thi & Đời sống sinh viên",
            "Số ca tin giả": 95,
            "Tỷ lệ tăng tuần": "-12.0%",
            "Mức độ cảnh báo": "🟢 AN TOÀN (Tương tác tích cực)",
            "Hành động đề xuất": "Bảo vệ các câu chuyện đẹp truyền cảm hứng (như UDA In Me)"
        }
    ]
    return pd.DataFrame(data)

def get_flagged_users_table() -> pd.DataFrame:
    """Bảng thống kê các tài khoản mạng xã hội bị gắn cờ rủi ro cao (User Flagging)."""
    data = [
        {
            "Tài khoản / Fanpage": "Việc Làm KCN Hòa Mạc 247",
            "Số bài đăng tin giả": 14,
            "Tỷ lệ Buff Like ảo": "88%",
            "Điểm uy tín (0-100)": 15,
            "Trạng thái tài khoản": "🚫 ĐÃ GẮN CỜ ĐEN (Blacklisted)",
            "Lý do vi phạm": "Mạo danh công ty in ấn Mintoku để lừa tiền đặt cọc"
        },
        {
            "Tài khoản / Fanpage": "Lương Y Truyền Đời Gia Truyền 100%",
            "Số bài đăng tin giả": 12,
            "Tỷ lệ Buff Like ảo": "92%",
            "Điểm uy tín (0-100)": 18,
            "Trạng thái tài khoản": "🚫 ĐÃ GẮN CỜ ĐEN (Blacklisted)",
            "Lý do vi phạm": "Quảng cáo nước lá trị dứt điểm ung thư không kiểm chứng"
        },
        {
            "Tài khoản / Fanpage": "Đầu Tư Tài Chính Sinh Lời 4.0",
            "Số bài đăng tin giả": 9,
            "Tỷ lệ Buff Like ảo": "76%",
            "Điểm uy tín (0-100)": 28,
            "Trạng thái tài khoản": "⚠️ CẢNH BÁO CAO (Theo dõi)",
            "Lý do vi phạm": "Rủ rê nạp tiền qua mã giới thiệu app giả mạo"
        },
        {
            "Tài khoản / Fanpage": "Phòng Trọ Đà Nẵng Giá Rẻ 2026",
            "Số bài đăng tin giả": 6,
            "Tỷ lệ Buff Like ảo": "65%",
            "Điểm uy tín (0-100)": 35,
            "Trạng thái tài khoản": "⚠️ CẢNH BÁO (Theo dõi)",
            "Lý do vi phạm": "Lấy ảnh biệt thự Nam Ô ghép vào bài cho thuê 4 triệu để câu cọc"
        },
        {
            "Tài khoản / Fanpage": "Mintoku Work Vietnam (Official)",
            "Số bài đăng tin giả": 0,
            "Tỷ lệ Buff Like ảo": "0%",
            "Điểm uy tín (0-100)": 98,
            "Trạng thái tài khoản": "✅ XÁC THỰC UY TÍN (Whitelisted)",
            "Lý do vi phạm": "Nhà tuyển dụng minh bạch, có phản hồi trực tiếp không mất phí"
        }
    ]
    return pd.DataFrame(data)

def get_regional_risk_table() -> pd.DataFrame:
    """Bảng phân tích rủi ro tin giả theo khu vực địa lý."""
    data = [
        {"Khu vực / Tỉnh thành": "TP. Hồ Chí Minh", "Số tin giả ghi nhận": 480, "Chủ đề nóng nhất": "Lừa đảo tài chính & Việc làm online", "Mức rủi ro": "🔴 Cao"},
        {"Khu vực / Tỉnh thành": "Hà Nội", "Số tin giả ghi nhận": 410, "Chủ đề nóng nhất": "Y tế, thuốc giả & Vé sự kiện", "Mức rủi ro": "🔴 Cao"},
        {"Khu vực / Tỉnh thành": "Đà Nẵng & Miền Trung", "Số tin giả ghi nhận": 230, "Chủ đề nóng nhất": "Phòng trọ ảo & Tin đồn bão lũ", "Mức rủi ro": "🟡 Trung bình"},
        {"Khu vực / Tỉnh thành": "Bình Dương / Đồng Nai", "Số tin giả ghi nhận": 195, "Chủ đề nóng nhất": "Tuyển dụng mạo danh KCN", "Mức rủi ro": "🟡 Trung bình"},
        {"Khu vực / Tỉnh thành": "Các tỉnh khác", "Số tin giả ghi nhận": 140, "Chủ đề nóng nhất": "Thần dược gia truyền, mê tín", "Mức rủi ro": "🟢 Thấp"}
    ]
    return pd.DataFrame(data)

if __name__ == "__main__":
    p1 = generate_time_series_chart()
    p2 = generate_hashtag_keyword_chart()
    print("Trend analysis charts ready:", p1, p2)
