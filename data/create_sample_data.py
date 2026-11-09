"""
Script tạo bộ dữ liệu mẫu (Sample Dataset) giả lập bài đăng mạng xã hội tiếng Việt.
Hỗ trợ cả văn bản có teencode, emoji và metadata (likes, shares, comments, is_verified).
Dùng để kiểm thử toàn bộ pipeline (Preprocessing -> Baseline -> PhoBERT -> Demo)
trước khi đưa tập dữ liệu lớn thực tế (VLSP ReINTEL / ViFactCheck) vào huấn luyện.
"""

import os
import sys
import random
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def generate_sample_data(num_samples: int = 120, output_path: str = "data/raw/data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Mẫu tin thật: tin thời sự, y tế chính thống, giáo dục, thời tiết
    real_templates = [
        "Bộ Y tế khuyến cáo người dân tuân thủ các biện pháp phòng chống dịch bệnh mùa hè.",
        "Bộ Giáo dục và Đào tạo công bố lịch thi tốt nghiệp THPT năm 2024 dự kiến vào cuối tháng 6.",
        "Trung tâm Dự báo Khí tượng Thủy văn Quốc gia cảnh báo đợt không khí lạnh tăng cường tại miền Bắc.",
        "Hôm nay, tuyến đường sắt đô thị Cát Linh - Hà Đông đã phục vụ hơn 30.000 lượt khách.",
        "Ngân hàng Nhà nước tiếp tục giữ nguyên lãi suất điều hành nhằm hỗ trợ doanh nghiệp phục hồi kinh tế.",
        "Việt Nam và đối tác quốc tế ký kết hợp tác phát triển năng lượng xanh và chuyển đổi số.",
        "Cục An toàn thông tin cảnh báo các chiêu trò lừa đảo mạo danh ngân hàng qua tin nhắn SMS Brandname.",
        "Bộ Giao thông Vận tải yêu cầu đẩy nhanh tiến độ thi công cao tốc Bắc - Nam giai đoạn 2.",
        "Các trường đại học bắt đầu mở cổng đăng ký xét tuyển học bạ đợt 1 cho tân sinh viên.",
        "Việt Nam đứng trong top các quốc gia có tốc độ tăng trưởng kinh tế số nhanh nhất Đông Nam Á."
    ]
    
    # Mẫu tin giả: giật gân, chữa bệnh thần tốc, tin đồn thất thiệt, câu tương tác, teencode
    fake_templates = [
        "BÀ CON CHÚ Ý: Uống nước lá ổi đun sôi cùng chanh sả cam kết trị dứt điểm 100% tế bào ung thư sau 7 ngày!!!",
        "KHẨN CẤP: Lộ clip mật máy bay rơi tại vùng biển Vũng Tàu, chính quyền đang phong tỏa giấu nhẹm tin tức???",
        "Sốc: Uống 2 cốc nước ngọt này mỗi ngày sẽ khiến não bộ ngừng hoạt động sau 1 tháng, chia sẻ gấp cho người thân!",
        "Ai đang dùng mạng 4G chú ý: Soạn tin XINTIEN gửi 1900xxxx để nhận ngay 500k tài khoản chính từ nhà mạng, e test r nhé mn ơi!",
        "Chấn động: Phát hiện sinh vật ngoài hành tinh dài 10m dạt vào bờ biển Đà Nẵng sáng nay, share ngay kẻo bị xóa bài!!",
        "CẢNH BÁO NGUY HIỂM: Ăn trứng gà cùng sữa đậu nành sẽ tạo ra chất kịch độc gây đột tử ngay lập tức!",
        "Tin mật từ chuyên gia: Giá vàng ngày mai sẽ tụt dốc về 20 triệu/lượng, ai có vàng bán tháo ngay lập tức kẻo phá sản!!",
        "Tập đoàn X tặng 10.000 xe máy điện nhân dịp kỷ niệm 20 năm, làm theo hướng dẫn: like + share + cmt số điện thoại để nhận quà!",
        "Tin chấn động giới showbiz: Ca sĩ A vừa bị bắt khẩn cấp tại sân bay vì buôn lậu, hình ảnh đang bị chặn!",
        "Bí kíp trúng số độc đắc 100%: Nhắn tin cho thầy B qua zalo 09xxxx để được phát lộc tâm linh, cam đoan uy tín!!"
    ]

    # Các từ nối hoặc biến thể mạng xã hội
    prefixes_real = ["Thông báo chính thức:", "Tin tức hôm nay:", "Bản tin thời sự:", "Theo TTXVN:", ""]
    prefixes_fake = ["BẤT NGỜ CHƯA:", "CHIA SẺ GẤP:", "CỰC SỐC:", "TIN NÓNG 24H:", "HOT HOT HOT:"]
    emojis_fake = [" 😱😱😱", " ❌❌❌", " 💥💥💥", " ⚡️⚡️", " 🚨🚨🚨"]
    emojis_real = [" ✅", " 📌", " 🌐", ""]

    data = []
    
    for i in range(num_samples // 2):
        # Tạo tin thật (Label = 0)
        base_text = random.choice(real_templates)
        pref = random.choice(prefixes_real)
        emo = random.choice(emojis_real)
        text = f"{pref} {base_text}{emo}".strip()
        likes = random.randint(50, 1500)
        shares = random.randint(5, 200)
        comments = random.randint(10, 300)
        is_verified = 1 if random.random() > 0.4 else 0  # Tài khoản chính thống có tick xanh cao hơn
        data.append({
            "id": f"real_{i+1:04d}",
            "text": text,
            "likes": likes,
            "shares": shares,
            "comments": comments,
            "is_verified": is_verified,
            "label": 0
        })

    for i in range(num_samples // 2):
        # Tạo tin giả (Label = 1)
        base_text = random.choice(fake_templates)
        pref = random.choice(prefixes_fake)
        emo = random.choice(emojis_fake)
        text = f"{pref} {base_text}{emo}".strip()
        likes = random.randint(300, 10000)      # Tin giả hay có like/share đột biến do giật gân
        shares = random.randint(100, 5000)
        comments = random.randint(50, 2000)
        is_verified = 1 if random.random() > 0.9 else 0  # Tài khoản tin giả rất ít khi có tick xanh
        data.append({
            "id": f"fake_{i+1:04d}",
            "text": text,
            "likes": likes,
            "shares": shares,
            "comments": comments,
            "is_verified": is_verified,
            "label": 1
        })

    random.shuffle(data)
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"-> Đã tạo tập dữ liệu mẫu thành công: {output_path} ({len(df)} dòng)")
    print(df["label"].value_counts())

if __name__ == "__main__":
    generate_sample_data(num_samples=200)
