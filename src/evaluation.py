"""
evaluation.py
-------------
Evaluation metrics and publication-quality visualisations for the
FT-Transformer diabetes classification project.

Functions:
  • plot_correlation_heatmap    — feature correlation matrix
  • plot_confusion_matrix       — heatmap of TP/TN/FP/FN
  • plot_training_history       — train & test loss curves
  • plot_auc_history            — per-epoch AUC curve
  • plot_rmse_history           — per-epoch RMSE curve
  • plot_roc_curves             — combined ROC for all models
  • plot_shap_summary           — SHAP beeswarm/bar plot
  • print_metrics_table         — pretty-print comparison DataFrame
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# Data / feature visualisations
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    exclude_col: str = "Outcome",
    save_path: str = None,
) -> None:
    """
    Plot a Pearson correlation heatmap for all numeric features,
    optionally excluding the target column.
    """
    corr = df.drop(columns=[exclude_col], errors="ignore").corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix (Excluding Target)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, format=save_path.split(".")[-1])
        print(f"[saved] {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def print_classification_report(y_true, y_pred, label: str = "Model") -> None:
    """Print precision / recall / F1 classification report."""
    print(f"\n{'='*60}")
    print(f"  Classification Report — {label}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=["No Diabetes", "Diabetes"]))


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> None:
    """Annotated heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
        annot_kws={"size": 16},
    )
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, format=save_path.split(".")[-1])
        print(f"[saved] {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Training history plots
# ---------------------------------------------------------------------------

def plot_training_history(
    train_loss: list,
    test_loss: list,
    title: str = "Training & Test Loss",
    save_path: str = None,
) -> None:
    """Line plot of loss curves across epochs."""
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, format=save_path.split(".")[-1])
        print(f"[saved] {save_path}")
    plt.show()


def plot_auc_history(
    auc_history: list,
    title: str = "Test AUC History",
    save_path: str = None,
) -> None:
    """Per-epoch AUC line plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(auc_history) + 1), auc_history, color="steelblue")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, format=save_path.split(".")[-1])
        print(f"[saved] {save_path}")
    plt.show()


def plot_rmse_history(
    rmse_history: list,
    title: str = "Test RMSE History",
    save_path: str = None,
) -> None:
    """Per-epoch RMSE line plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rmse_history) + 1), rmse_history, color="tomato")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, format=save_path.split(".")[-1])
        print(f"[saved] {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# ROC curve comparison
# ---------------------------------------------------------------------------

# Colour / style scheme for each model
ROC_STYLE = {
    "Logistic Regression": dict(color="blue",   lw=1, linestyle="--"),
    "XGBoost":             dict(color="green",  lw=1, linestyle="-."),
    "Random Forest":       dict(color="orange", lw=1, linestyle="-"),
    "TabNet":              dict(color="purple", lw=1, linestyle="dotted"),
    "LightGBM":            dict(color="brown",  lw=1, linestyle=":"),
    "FTTransformer":       dict(color="red",    lw=1, marker="*"),
}


def plot_roc_curves(
    roc_data: dict,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
    save_path: str = None,
) -> None:
    """
    Combined ROC plot for all models.

    Parameters
    ----------
    roc_data : dict  name → {"fpr": ..., "tpr": ..., "auc": float}
    """
    plt.figure(figsize=(8, 6))
    for name, data in roc_data.items():
        style = ROC_STYLE.get(name, dict(lw=1))
        plt.plot(
            data["fpr"],
            data["tpr"],
            label=f"{name} (AUC = {data['auc']:.4f})",
            **style,
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, format=save_path.split(".")[-1])
        print(f"[saved] {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# SHAP / LIME interpretability
# ---------------------------------------------------------------------------

def plot_shap_summary(shap_values, X_test_combined, feature_names: list, save_path: str = None) -> None:
    """
    SHAP beeswarm summary plot.
    Requires `shap` to be imported by the caller.
    """
    import shap
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_combined, feature_names=feature_names, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_metrics_table(results_df: pd.DataFrame) -> None:
    """Pretty-print the model comparison DataFrame."""
    print("\n" + "=" * 70)
    print("  Model Comparison Summary")
    print("=" * 70)
    print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 70 + "\n")
