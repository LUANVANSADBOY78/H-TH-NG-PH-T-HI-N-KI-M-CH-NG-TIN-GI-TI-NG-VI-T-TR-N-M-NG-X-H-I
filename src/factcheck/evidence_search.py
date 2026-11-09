"""
Module Tìm kiếm & Đối chiếu Bằng chứng (Evidence Search & Retrieval Engine).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.database.connection import get_connection

def search_matching_evidence(claim_text: str, top_k: int = 2) -> list:
    """Truy xuất bằng chứng có độ tương đồng cao nhất từ kho tri thức."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content, verdict, publisher, source_url FROM evidence_sources")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    evidences = [dict(r) for r in rows]
    corpus = [claim_text] + [f"{e['title']} {e['content']}" for e in evidences]

    vec = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vec.fit_transform(corpus)
    similarities = cosine_similarity(matrix[0], matrix[1:]).flatten()

    top_idx = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_idx:
        sim = float(similarities[idx])
        if sim > 0.08:
            ev = evidences[idx]
            ev["similarity"] = round(sim, 3)
            results.append(ev)
    return results

