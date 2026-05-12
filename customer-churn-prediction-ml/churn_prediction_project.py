"""
CSC-44112 Part 2: Predicting Customer Churn Using Explainable Machine Learning Techniques

Run:
    python churn_prediction_project.py

Dataset file required in the same folder:
    WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_clean_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_FILE}. "
            "Download it from Kaggle and place it in this project folder."
        )

    df = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")
    print("Original shape:", df.shape)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print("Cleaned shape:", df.shape)
    return df


def generate_eda_figures(df):
    plt.figure(figsize=(10, 5))
    plt.imshow(df.isnull(), aspect="auto")
    plt.title("Figure 1: Missing Value Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Customer Records")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure1_missing_value_heatmap.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    df["Churn"].value_counts().sort_index().plot(kind="bar")
    plt.xticks([0, 1], ["No Churn", "Churn"], rotation=0)
    plt.title("Figure 2: Churn Distribution")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure2_churn_distribution.png", dpi=300)
    plt.close()

    contract_counts = pd.crosstab(df["Contract"], df["Churn"])
    contract_counts.plot(kind="bar", figsize=(8, 5))
    plt.title("Figure 3: Contract Type versus Churn")
    plt.xlabel("Contract Type")
    plt.ylabel("Customer Count")
    plt.legend(["No Churn", "Churn"])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure3_contract_vs_churn.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df[df["Churn"] == 0]["tenure"], bins=30, alpha=0.7, label="No Churn")
    plt.hist(df[df["Churn"] == 1]["tenure"], bins=30, alpha=0.7, label="Churn")
    plt.title("Figure 4: Tenure Distribution by Churn")
    plt.xlabel("Tenure")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure4_tenure_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    df.boxplot(column="MonthlyCharges", by="Churn")
    plt.title("Figure 5: Monthly Charges by Churn")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure5_monthly_charges.png", dpi=300)
    plt.close()

    print("EDA figures generated.")


def preprocess_data(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X, X_train_scaled, X_test_scaled, y_train, y_test


def train_and_evaluate_models(X, X_train_scaled, X_test_scaled, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        )
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]

        results.append([
            name,
            accuracy_score(y_test, predictions),
            precision_score(y_test, predictions),
            recall_score(y_test, predictions),
            f1_score(y_test, predictions),
            roc_auc_score(y_test, probabilities)
        ])

        print(f"\n{name} Classification Report")
        print(classification_report(y_test, predictions))

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    )

    results_df.to_csv(f"{OUTPUT_DIR}/model_comparison_results.csv", index=False)
    print(results_df.round(3))

    return models, results_df


def generate_evaluation_figures(X, X_test_scaled, y_test, models, results_df):
    plt.figure(figsize=(9, 5))
    plt.bar(results_df["Model"], results_df["ROC-AUC"])
    plt.title("Figure 6: Model ROC-AUC Comparison")
    plt.ylabel("ROC-AUC")
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure6_model_roc_auc_comparison.png", dpi=300)
    plt.close()

    xgb_model = models["XGBoost"]
    xgb_predictions = xgb_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, xgb_predictions)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Figure 7: XGBoost Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["No Churn", "Churn"])
    plt.yticks([0, 1], ["No Churn", "Churn"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure7_xgboost_confusion_matrix.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.plot(fpr, tpr, label=name)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.title("Figure 8: ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure8_roc_curve_comparison.png", dpi=300)
    plt.close()

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": xgb_model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    ).head(12)

    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )
    plt.gca().invert_yaxis()
    plt.title("Figure 9: XGBoost Feature Importance Explainability")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure9_xgboost_feature_importance.png", dpi=300)
    plt.close()

    feature_importance.to_csv(
        f"{OUTPUT_DIR}/xgboost_feature_importance.csv",
        index=False
    )

    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        warm_start=True,
        random_state=42
    )

    losses = []

    X_train_again, _, y_train_again, _ = train_test_split(
        X_test_scaled,
        y_test,
        test_size=0.3,
        random_state=42,
        stratify=y_test
    )

    for epoch in range(50):
        nn.fit(X_train_again, y_train_again)
        losses.append(nn.loss_)

    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("Figure 10: Neural Network Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure10_neural_network_learning_curve.png", dpi=300)
    plt.close()

    print("Evaluation figures generated.")


def main():
    df = load_and_clean_data()
    generate_eda_figures(df)
    X, X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)
    models, results_df = train_and_evaluate_models(
        X,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )
    generate_evaluation_figures(X, X_test_scaled, y_test, models, results_df)
    print(f"Project completed. Outputs saved in '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    main()
