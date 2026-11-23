"""
Xây dựng mô hình Baseline truyền thống (TF-IDF + Các thuật toán Machine Learning).
Đo kiểm thực nghiệm chuẩn xác trên tập Kiểm thử Độc lập (Test Set: 390 mẫu).
Bao gồm:
1. Logistic Regression
2. Linear Support Vector Machine (LinearSVC)
3. Multinomial Naive Bayes (MultinomialNB)
4. Random Forest Classifier
"""

import os
import sys
import json
import pandas as pd
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PROCESSED_DATA_DIR, EVALUATION_DIR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def run_baselines(data_dir: str = PROCESSED_DATA_DIR, output_dir: str = EVALUATION_DIR):
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Chưa có dữ liệu trong {data_dir}. Hãy chạy build_complete_dataset.py trước!")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    text_col = "text_segmented" if "text_segmented" in train_df.columns else ("text_seg" if "text_seg" in train_df.columns else "text")
    X_train, y_train = train_df[text_col].astype(str), train_df["label"].astype(int)
    X_test, y_test = test_df[text_col].astype(str), test_df["label"].astype(int)

    n_test_0 = int(np.sum(y_test == 0))
    n_test_1 = int(np.sum(y_test == 1))

    print("=" * 68)
    print(f">>> HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH BASELINE TRÊN TEST SET (N={len(y_test)}) <<<")
    print(f"[*] Phân bố tập Test: Lớp 0 (Tin Thật) = {n_test_0} | Lớp 1 (Tin Giả) = {n_test_1}")
    print("=" * 68)

    models = {
        "Logistic Regression (TF-IDF)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Linear SVM (TF-IDF)": LinearSVC(class_weight="balanced", random_state=42, dual="auto"),
        "Multinomial Naive Bayes (TF-IDF)": MultinomialNB(),
        "Random Forest (TF-IDF)": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    }

    results = []
    baseline_export = {}

    for name, clf in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=10000,
                sublinear_tf=True
            )),
            ("clf", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\n=== {name} ===")
        print(f"Accuracy : {acc:.4f} | Macro F1: {macro_f1:.4f} | Precision: {macro_prec:.4f} | Recall: {macro_rec:.4f}")
        print("Confusion Matrix:")
        print(cm)

        res_dict = {
            "model_name": name,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "confusion_matrix": {
                "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "row_sums": [int(tn + fp), int(fn + tp)]
            }
        }
        results.append(res_dict)
        baseline_export[name] = res_dict

    out_file = os.path.join(output_dir, "baseline_metrics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(baseline_export, f, indent=2, ensure_ascii=False)
    print(f"\n[✔] Đã lưu kết quả thực nghiệm Baselines tại: {out_file}")

    summary_df = pd.DataFrame([
        {"Mô hình": r["model_name"], "Accuracy": r["accuracy"], "Macro F1": r["macro_f1"], "Precision": r["precision"], "Recall": r["recall"]}
        for r in results
    ])
    print("\n--- BẢNG TỔNG HỢP KẾT QUẢ BASELINES TRÊN TEST SET (N=390) ---")
    print(summary_df.to_string(index=False))
    return summary_df

if __name__ == "__main__":
    run_baselines()
