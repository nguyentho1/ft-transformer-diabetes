"""
main.py
-------
End-to-end pipeline for diabetes classification using FT-Transformer
and five baseline models.

Usage
-----
    python main.py --data data/raw/frankfurt_diabetes.csv

Steps
-----
  1. Preprocess data (clean, impute, scale, split)
  2. Train FT-Transformer (800 epochs, lr=0.001, early stopping)
  3. Train baseline models (LR, RF, XGBoost, LightGBM, TabNet)
  4. Evaluate all models and display comparison table
  5. Generate and save visualisations to outputs/
"""

import argparse
import os

import torch

from src.data_preprocessing import (
    run_preprocessing,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.ft_transformer import build_dataloaders
from src.train import build_model, train_ft_transformer, DEFAULT_CONFIG
from src.baseline_models import (
    train_sklearn_baselines,
    train_tabnet,
    compute_roc_curves,
    build_results_table,
    _metrics,
)
from src.evaluation import (
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_training_history,
    plot_auc_history,
    plot_rmse_history,
    plot_roc_curves,
    print_classification_report,
    print_metrics_table,
)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="FT-Transformer Diabetes Classification")
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/frankfurt_diabetes.csv",
        help="Path to the raw CSV dataset",
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Preprocessing
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 1: Data Preprocessing")
    print("=" * 60)
    X_train, X_test, y_train, y_test, scaler, df_processed = run_preprocessing(args.data)

    # Correlation heatmap of processed features
    plot_correlation_heatmap(df_processed, save_path="outputs/correlation_heatmap.eps")

    # ------------------------------------------------------------------
    # 2. FT-Transformer
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 2: FT-Transformer Training")
    print("=" * 60)

    categorical_cardinalities = [df_processed["Pregnancies"].nunique()]
    ft_config = dict(
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
    )

    model = build_model(categorical_cardinalities, ft_config)
    train_loader, test_loader = build_dataloaders(
        X_train, X_test, y_train, y_test,
        NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        batch_size=args.batch_size,
    )

    ft_results = train_ft_transformer(
        model, train_loader, test_loader, X_test, y_test, ft_config
    )

    # Save model checkpoint
    torch.save(model.state_dict(), "outputs/ft_transformer_weights.pt")
    print("Model weights saved → outputs/ft_transformer_weights.pt")

    # Training history plots
    plot_training_history(
        ft_results["train_loss_history"],
        ft_results["test_loss_history"],
        save_path="outputs/loss_history.eps",
    )
    plot_auc_history(ft_results["auc_history"], save_path="outputs/auc_history.eps")
    plot_rmse_history(ft_results["rmse_history"], save_path="outputs/rmse_history.eps")

    # Confusion matrix
    plot_confusion_matrix(
        y_test, ft_results["y_pred"],
        title="FT-Transformer Confusion Matrix",
        save_path="outputs/ft_confusion_matrix.eps",
    )
    print_classification_report(y_test, ft_results["y_pred"], label="FT-Transformer")

    ft_metric = _metrics(
        y_test, ft_results["y_pred"], ft_results["test_time"], "FTTransformer"
    )

    # ------------------------------------------------------------------
    # 3. Baseline models
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 3: Baseline Model Training")
    print("=" * 60)

    classifiers, sklearn_results = train_sklearn_baselines(X_train, X_test, y_train, y_test)
    tabnet_model, tabnet_result = train_tabnet(X_train, X_test, y_train, y_test)

    # ------------------------------------------------------------------
    # 4. Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 4: Model Comparison")
    print("=" * 60)

    results_df = build_results_table(sklearn_results, tabnet_result, ft_metric)
    print_metrics_table(results_df)
    results_df.to_csv("outputs/model_comparison.csv")
    print("Results saved → outputs/model_comparison.csv")

    # ------------------------------------------------------------------
    # 5. ROC curves
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 5: ROC Curves")
    print("=" * 60)

    roc_data = compute_roc_curves(
        classifiers, tabnet_model, model,
        X_test, y_test,
        NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    )
    plot_roc_curves(roc_data, save_path="outputs/roc_curves.eps")

    print("\n Pipeline complete. All outputs saved to outputs/")


if __name__ == "__main__":
    main()
