import os
import json
import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import xgboost as xgb

def run_tournament():
    print("[INFO] ========================================================")
    print("[INFO] FULL DATASET MULTI-MODEL TOURNAMENT & ERROR ANALYSIS")
    print("[INFO] ========================================================")

    processed_dir = "data/processed"
    models_dir = "models"
    artifacts_dir = "artifacts"
    outputs_dir = "outputs"
    reports_dir = "reports"

    for d in [models_dir, artifacts_dir, outputs_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    X_train = np.load(os.path.join(processed_dir, "X_train_final.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test_final.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    X_test_raw = pd.read_csv(os.path.join(processed_dir, "X_test_raw.csv"))

    print(f"[INFO] Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]} | Features: {X_train.shape[1]}")

    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "file": "logistic_regression_baseline.pkl"
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
            "file": "decision_tree_baseline.pkl"
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight={0: 1.0, 1: 2.5}),
            "file": "random_forest_baseline.pkl"
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(scale_pos_weight=2.5, max_depth=4, learning_rate=0.05, n_estimators=100, eval_metric="logloss", random_state=42),
            "file": "xgboost_model.pkl"
        }
    }

    results = {}
    roc_curves = {}

    best_model_name = None
    best_recall = -1.0
    best_model_obj = None

    for name, item in models.items():
        clf = item["model"]
        print(f"\n[TRAIN] Fitting {name}...")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

        roc_curves[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

        model_save_path = os.path.join(models_dir, item["file"])
        joblib.dump(clf, model_save_path)
        print(f"       -> Saved to {model_save_path}")
        print(f"       -> Recall: {rec:.4f} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")

        if rec > best_recall:
            best_recall = rec
            best_model_name = name
            best_model_obj = clf

    print("\n" + "=" * 60)
    print(f"[CHAMPION] Best Model by Recall: {best_model_name} (Recall: {best_recall:.4f})")
    print("=" * 60)

    # 1. Plot and save Multi-Model ROC curves
    plt.figure(figsize=(9, 6))
    for name, roc_data in roc_curves.items():
        plt.plot(roc_data["fpr"], roc_data["tpr"], label=f"{name} (AUC = {roc_data['auc']:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5000)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC-AUC Comparison Across All Models (Held-out Test Split)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    roc_plot_path = os.path.join(artifacts_dir, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Comparative ROC curves saved to: {roc_plot_path}")

    # 2. Plot and save Confusion Matrix for the best model
    champion_preds = best_model_obj.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        champion_preds,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        ax=ax
    )
    plt.title(f"Confusion Matrix - Champion Model: {best_model_name}")
    plt.tight_layout()
    cm_plot_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Champion Confusion Matrix saved to: {cm_plot_path}")

    # 3. Perform Error Analysis on the entire test set
    error_analysis_df = X_test_raw.copy()
    error_analysis_df["Actual_Churn"] = y_test
    error_analysis_df["Predicted_Churn"] = champion_preds
    error_analysis_df["Churn_Probability"] = np.round(best_model_obj.predict_proba(X_test)[:, 1], 4)

    # False Negatives: Actual = 1, Predicted = 0 (Missed churners)
    false_negatives = error_analysis_df[
        (error_analysis_df["Actual_Churn"] == 1) & (error_analysis_df["Predicted_Churn"] == 0)
    ]
    fn_path = os.path.join(outputs_dir, "false_negatives.csv")
    false_negatives.to_csv(fn_path, index=False)
    print(f"[SUCCESS] False Negatives saved ({len(false_negatives)} cases) to: {fn_path}")

    # False Positives: Actual = 0, Predicted = 1 (False alarms)
    false_positives = error_analysis_df[
        (error_analysis_df["Actual_Churn"] == 0) & (error_analysis_df["Predicted_Churn"] == 1)
    ]
    fp_path = os.path.join(outputs_dir, "false_positives.csv")
    false_positives.to_csv(fp_path, index=False)
    print(f"[SUCCESS] False Positives saved ({len(false_positives)} cases) to: {fp_path}")

    # 4. Save Comprehensive Tournament Report
    tournament_report = {
        "tournament_name": "Telco Customer Churn Multi-Model Tournament",
        "dataset_total_records": 7344,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features_count": int(X_train.shape[1]),
        "champion_model": best_model_name,
        "champion_recall": best_recall,
        "models_summary": results,
        "error_analysis": {
            "total_test_records": int(len(y_test)),
            "false_negatives_count": int(len(false_negatives)),
            "false_positives_count": int(len(false_positives)),
            "true_positives_count": int(results[best_model_name]["confusion_matrix"][1][1]),
            "true_negatives_count": int(results[best_model_name]["confusion_matrix"][0][0]),
        }
    }

    report_path = os.path.join(reports_dir, "model_tournament_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(tournament_report, f, indent=4)
    print(f"[SUCCESS] Model tournament report saved to: {report_path}")

    # Also save champion model to models/churn_model.pkl
    champion_path = os.path.join(models_dir, "churn_model.pkl")
    joblib.dump(best_model_obj, champion_path)
    print(f"[SUCCESS] Champion model saved as primary model to: {champion_path}")

    print("\n[COMPLETE] Full dataset tournament and error analysis finished successfully!")

if __name__ == "__main__":
    run_tournament()

