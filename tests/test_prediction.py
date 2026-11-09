"""
Unit tests for Fact-checking & Prediction modules.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.factcheck.source_check import check_domain_credibility
from src.factcheck.evidence_search import search_matching_evidence
from src.prediction.explain import explain_text

class TestPrediction(unittest.TestCase):
    def test_domain_credibility(self):
        res = check_domain_credibility("https://vtv.vn/thoi-su.htm")
        self.assertTrue(res["is_official"])
        self.assertGreaterEqual(res["trust_score"], 90)

    def test_social_domain(self):
        res = check_domain_credibility("https://www.facebook.com/share/p/123/")
        self.assertTrue(res["is_social"])

    def test_xai_explain(self):
        text = "Khẩn cấp: Uống nước lá ổi cam kết trị dứt điểm ung thư 100%!"
        res = explain_text(text, "Refuted")
        self.assertTrue(res["is_sensational"])
        self.assertGreaterEqual(res["clickbait_score"], 50)

if __name__ == "__main__":
    unittest.main()
