import sys
import os
import unittest
import pandas as pd

# Setup paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)
sys.path.insert(0, os.path.join(BASE_PATH, "src"))

from main import handle_verify_article, handle_extract_url
from src.batch_analyzer import process_batch_file
from src.advanced_nlp import AdvancedNLPToolkit
from src.chatbot import FactCheckChatbot
from src.database import save_history, get_history
from src.admin.users import get_all_users
from src.admin.model_manager import get_all_model_versions
from src.admin.dataset_manager import get_registered_datasets
from src.ablation_and_metrics import (
    get_ablation_study_table,
    generate_training_loss_plot,
    generate_confusion_matrix_plot
)

class FullSystemTestCase(unittest.TestCase):
    def test_01_tab1_fake_medical(self):
        text = "Bà con chú ý: uống nước lá ổi tươi chữa dứt điểm 100% bệnh ung thư kịch độc!"
        res = handle_verify_article(
            headline="Tin y tế lá ổi",
            text=text,
            url="",
            image_input=None,
            post_time="Vừa đăng gần đây (Hôm nay)",
            enable_chunking=True,
            likes=10,
            shares=5,
            comments=2,
            is_verified=False,
            fb_active_daily=True,
            fb_messages_daily=True
        )
        self.assertEqual(len(res), 13)
        self.assertIn("TIN GIẢ", res[0])
        self.assertGreater(res[1]["🔴 Tin Giả / Nguy Cơ Lừa Đảo"], 0.7)

    def test_02_tab1_real_student(self):
        text = '✨ "LẦN ĐẦU TÔI DÁM ƯỚC MƠ – UDA IN ME" ✨ Có những ước mơ từng nằm yên trong tôi suốt nhiều năm tại Đại học Đông Á...'
        res = handle_verify_article(
            headline="",
            text=text,
            url="",
            image_input=None,
            post_time="Vừa đăng gần đây (Hôm nay)",
            enable_chunking=True,
            likes=0,
            shares=0,
            comments=0,
            is_verified=False,
            fb_active_daily=True,
            fb_messages_daily=True
        )
        self.assertEqual(len(res), 13)
        self.assertIn("NỘI DUNG THẬT", res[0])
        self.assertGreater(res[1]["🟢 Tin Thật / Nội Dung An Toàn"], 0.8)

    def test_03_tab1_empty_text(self):
        res = handle_verify_article(
            headline="",
            text="",
            url="",
            image_input=None,
            post_time="Vừa đăng gần đây (Hôm nay)",
            enable_chunking=True,
            likes=0,
            shares=0,
            comments=0,
            is_verified=False,
            fb_active_daily=True,
            fb_messages_daily=True
        )
        self.assertEqual(len(res), 13)
        self.assertIn("Vui lòng nhập", res[0])

    def test_04_tab2_batch_inference(self):
        sample_csv = os.path.join(BASE_PATH, "outputs", "sample_batch_test.csv")
        self.assertTrue(os.path.exists(sample_csv))
        preview, out_path, msg = process_batch_file(sample_csv)
        self.assertIsNotNone(preview)
        self.assertTrue(os.path.exists(out_path))
        self.assertIn("thành công", msg)

    def test_05_tab3_text_compare(self):
        toolkit = AdvancedNLPToolkit()
        txt_a = "Bộ Y tế khuyến cáo đeo khẩu trang phòng dịch cúm mùa."
        txt_b = "Khuyến cáo người dân thực hiện các biện pháp y tế phòng dịch bệnh lây lan."
        cmp_res = toolkit.compare_two_articles(txt_a, txt_b)
        self.assertIn("similarity_score", cmp_res)
        self.assertGreaterEqual(cmp_res["similarity_score"], 0)

    def test_06_tab4_history_database(self):
        hist_id = save_history(
            content="Tin kiểm thử hệ thống tự động",
            predicted_label="🟢 NỘI DUNG THẬT",
            confidence=0.95,
            prob_real=0.95,
            prob_fake=0.05,
            prob_nei=0.0,
            source_url="test.vn",
            domain="test.vn",
            evidence_matched="Bằng chứng kiểm thử",
            user_name="admin"
        )
        self.assertIsNotNone(hist_id)
        df_hist = get_history(limit=5)
        self.assertGreater(len(df_hist), 0)

    def test_07_tab5_chatbot(self):
        bot = FactCheckChatbot()
        response = bot.respond("Chào bạn, bạn có thể kiểm tra tin tức gì?", history=[])
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 10)

    def test_08_tab6_admin_metrics(self):
        users_df = get_all_users()
        self.assertGreater(len(users_df), 0)
        models_df = get_all_model_versions()
        self.assertGreater(len(models_df), 0)
        datasets_df = get_registered_datasets()
        self.assertGreater(len(datasets_df), 0)
        ablation_df = get_ablation_study_table()
        self.assertGreater(len(ablation_df), 0)
        
        loss_plot = generate_training_loss_plot()
        self.assertTrue(os.path.exists(loss_plot))
        cm_plot = generate_confusion_matrix_plot()
        self.assertTrue(os.path.exists(cm_plot))

    def test_09_trend_analysis(self):
        from src.trend_analysis import (
            generate_time_series_chart,
            generate_hashtag_keyword_chart,
            get_topic_alert_summary,
            get_flagged_users_table,
            get_regional_risk_table
        )
        p1 = generate_time_series_chart()
        self.assertTrue(os.path.exists(p1))
        p2 = generate_hashtag_keyword_chart()
        self.assertTrue(os.path.exists(p2))
        t1 = get_topic_alert_summary()
        self.assertGreater(len(t1), 0)
        t2 = get_flagged_users_table()
        self.assertGreater(len(t2), 0)
        t3 = get_regional_risk_table()
        self.assertGreater(len(t3), 0)

if __name__ == "__main__":
    unittest.main()
