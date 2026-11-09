"""
Module Làm sạch & Tiền xử lý Dữ liệu Ngôn ngữ Mạng Xã Hội Tiếng Việt.
"""

import re
import unicodedata
import emoji
from underthesea import word_tokenize

TEENCODE_DICT = {
    "ko": "không", "k": "không", "kh": "không", "hem": "không", "hông": "không",
    "dc": "được", "dk": "được", "đc": "được",
    "mn": "mọi người", "mng": "mọi người",
    "r": "rồi", "rùi": "rồi",
    "j": "gì", "gi": "gì",
    "wa": "quá", "wá": "quá",
    "thui": "thôi",
    "bit": "biết", "bik": "biết",
    "nt": "nhắn tin", "ib": "nhắn tin", "inbox": "nhắn tin",
    "cx": "cũng",
    "vs": "với", "zới": "với",
    "ng": "người", "ngừi": "người",
    "bn": "bạn", "b": "bạn",
    "cmt": "bình luận", "share": "chia sẻ", "fb": "facebook",
    "vcl": "rất", "vđ": "rất", "vl": "rất"
}

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", str(text))

def replace_teencode(text: str) -> str:
    words = text.split()
    return " ".join([TEENCODE_DICT.get(w.lower(), w) for w in words])

def clean_text(text: str, remove_emoji: bool = True) -> str:
    text = normalize_unicode(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\b0\d{9,10}\b", " sdt_lien_he ", text)
    text = replace_teencode(text)
    if remove_emoji:
        text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def segment_text(text: str) -> str:
    try:
        return word_tokenize(text, format="text")
    except Exception:
        return text

