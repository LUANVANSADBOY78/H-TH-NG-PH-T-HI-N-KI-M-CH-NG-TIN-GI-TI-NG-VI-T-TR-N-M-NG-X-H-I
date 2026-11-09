"""
Module Kiểm tra Uy tín Nguồn tin & Tên miền (Source Credibility Checker).
"""

from urllib.parse import urlparse

OFFICIAL_WHITELIST = [
    "vtv.vn", "tuoitre.vn", "vnexpress.net", "thanhnien.vn", "dantri.com.vn",
    "vietnamnet.vn", "nhandan.vn", "moh.gov.vn", "chinhphu.vn", "baochinhphu.vn",
    "cand.com.vn", "qdnd.vn", "ttxvn.vn", "mic.gov.vn", "moet.gov.vn", "who.int"
]

SOCIAL_PLATFORMS = [
    "facebook.com", "fb.com", "tiktok.com", "youtube.com", "youtu.be",
    "threads.net", "twitter.com", "x.com", "instagram.com", "zalo.me"
]

def check_domain_credibility(url_or_domain: str) -> dict:
    domain = url_or_domain.lower().strip()
    if domain.startswith(("http://", "https://")):
        domain = urlparse(domain).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    is_official = any(white in domain for white in OFFICIAL_WHITELIST)
    is_social = any(soc in domain for soc in SOCIAL_PLATFORMS)

    if is_official:
        level = "Nguồn tin chính thống cấp nhà nước / Báo chí uy tín cao"
        trust_score = 95
    elif is_social:
        level = "Mạng xã hội (Thông tin tự do, cần kiểm chứng đối chiếu)"
        trust_score = 45
    else:
        level = "Trang tin độc lập / Chưa xác minh danh tính"
        trust_score = 60

    return {
        "domain": domain,
        "is_official": is_official,
        "is_social": is_social,
        "credibility_level": level,
        "trust_score": trust_score
    }

