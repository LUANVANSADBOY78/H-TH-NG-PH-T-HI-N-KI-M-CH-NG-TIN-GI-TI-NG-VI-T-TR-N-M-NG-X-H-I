"""
Xây dựng mô hình Baseline truyền thống (TF-IDF + Các thuật toán Machine Learning).
Baseline là cột mốc chuẩn để so sánh và chứng minh tính vượt trội của PhoBERT/Transformer.
Bao gồm:
1. Logistic Regression
2. Linear Support Vector Machine (LinearSVC)
3. Naive Bayes (MultinomialNB)
4. Random Forest Classifier
"""

import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

def run_baselines(data_dir: str = "data/processed"):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Chưa có dữ liệu tiền xử lý trong {data_dir}. Hãy chạy preprocess.py trước!")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Lấy văn bản đã tách từ hoặc văn bản sạch
    text_col = "text_segmented" if "text_segmented" in train_df.columns else ("text_seg" if "text_seg" in train_df.columns else "text")
    X_train, y_train = train_df[text_col].astype(str), train_df["label"].astype(int)
    X_val, y_val = val_df[text_col].astype(str), val_df["label"].astype(int)

    print(f"Số lượng tập huấn luyện: {len(X_train)} | Tập kiểm định (Val): {len(X_val)}")
    print("-" * 65)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42, dual="auto"),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    }

    results = []

    for name, clf in models.items():
        # Xây dựng Pipeline: TF-IDF (1-gram và 2-gram) kết hợp bộ phân lớp
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
        y_pred = pipeline.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro")

        print(f"=== {name} ===")
        print(f"Accuracy : {acc:.4f} | Macro F1: {macro_f1:.4f}")
        print("Chi tiết:")
        print(classification_report(y_val, y_pred, target_names=["Tin thật (0)", "Tin giả (1)"], digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("-" * 65)

        results.append({
            "Mô hình": name,
            "Accuracy": round(acc, 4),
            "Macro F1": round(macro_f1, 4)
        })

    summary_df = pd.DataFrame(results)
    print("\n--- BẢNG TỔNG KẾT BASELINE TRÊN TẬP VALIDATION ---")
    print(summary_df.to_string(index=False))
    return summary_df

if __name__ == "__main__":
    run_baselines()
