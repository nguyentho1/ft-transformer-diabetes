"""
baseline_models.py
------------------
Trains and evaluates five baseline classifiers for comparison against
the FT-Transformer:

  1. Logistic Regression
  2. Random Forest
  3. XGBoost
  4. LightGBM
  5. TabNet

Returns a summary DataFrame with Accuracy, Precision, Recall, F1, and
Test Inference Time for every model.
"""

import time

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import lightgbm as lgb
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timed_predict(clf, X_test) -> tuple[np.ndarray, float]:
    """Return predictions and the wall-clock inference time in seconds."""
    start = time.time()
    preds = clf.predict(X_test)
    return preds, time.time() - start


def _metrics(y_true, y_pred, test_time: float, model_name: str) -> dict:
    return dict(
        Model=model_name,
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        F1_Score=f1_score(y_true, y_pred, zero_division=0),
        Test_Time_s=test_time,
    )


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------

def train_sklearn_baselines(X_train, X_test, y_train, y_test) -> tuple[dict, list]:
    """
    Train Logistic Regression, Random Forest, XGBoost, and LightGBM.

    Returns
    -------
    classifiers : dict  name → fitted estimator
    results     : list of metric dicts
    """
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
    }

    results = []
    for name, clf in classifiers.items():
        print(f"Training {name}…")
        clf.fit(X_train, y_train)
        preds, t = _timed_predict(clf, X_test)
        results.append(_metrics(y_test, preds, t, name))
        print(f"  Accuracy: {results[-1]['Accuracy']:.4f}")

    return classifiers, results


def train_tabnet(X_train, X_test, y_train, y_test) -> tuple[TabNetClassifier, dict]:
    """
    Train a TabNetClassifier.

    Data must be NumPy float32 arrays (convert with .values.astype(np.float32)).
    """
    print("Training TabNet…")

    # Ensure numpy arrays with correct dtype
    Xtr = X_train if isinstance(X_train, np.ndarray) else X_train.values.astype(np.float32)
    Xte = X_test  if isinstance(X_test,  np.ndarray) else X_test.values.astype(np.float32)
    ytr = y_train.values if hasattr(y_train, "values") else y_train
    yte = y_test.values  if hasattr(y_test,  "values") else y_test

    tabnet = TabNetClassifier(
        n_d=64, n_a=64, n_steps=3, gamma=1.5,
        n_independent=2, n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.001),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 50, "gamma": 0.9},
        mask_type="entmax",
        verbose=0,
    )

    tabnet.fit(
        Xtr, ytr,
        eval_set=[(Xte, yte)],
        max_epochs=400,
        patience=20,
        batch_size=128,
        virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False,
    )

    preds, t = _timed_predict(tabnet, Xte)
    result = _metrics(yte, preds, t, "TabNet")
    print(f"  Accuracy: {result['Accuracy']:.4f}")
    return tabnet, result


# ---------------------------------------------------------------------------
# ROC curve computation
# ---------------------------------------------------------------------------

def compute_roc_curves(
    classifiers: dict,
    tabnet_model,
    ft_model,
    X_test,
    y_test,
    numeric_features: list,
    categorical_features: list,
) -> dict:
    """
    Compute (fpr, tpr, auc) for every model.

    Parameters
    ----------
    classifiers : dict  name → sklearn estimator (must have predict_proba)
    tabnet_model : TabNetClassifier
    ft_model : FTTransformer
    X_test, y_test : test data
    numeric_features, categorical_features : feature name lists for FT-Transformer

    Returns
    -------
    dict  name → {"fpr": ..., "tpr": ..., "auc": ...}
    """
    roc_data = {}

    # sklearn / LightGBM / XGBoost classifiers
    for name, clf in classifiers.items():
        proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(y_test, proba)}

    # TabNet
    Xte = X_test if isinstance(X_test, np.ndarray) else X_test.values.astype(np.float32)
    proba_tn = tabnet_model.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba_tn)
    roc_data["TabNet"] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(y_test, proba_tn)}

    # FT-Transformer
    ft_model.eval()
    with torch.no_grad():
        num_t = torch.tensor(X_test[numeric_features].values.astype("float32"))
        cat_t = torch.tensor(X_test[categorical_features].values.astype("int64"))
        proba_ft = ft_model(num_t, cat_t).squeeze().numpy()
    fpr, tpr, _ = roc_curve(y_test, proba_ft)
    roc_data["FTTransformer"] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(y_test, proba_ft)}

    return roc_data


def build_results_table(sklearn_results: list, tabnet_result: dict, ft_result: dict) -> pd.DataFrame:
    """Combine all metric dicts into a single comparison DataFrame."""
    all_results = sklearn_results + [tabnet_result, ft_result]
    df = pd.DataFrame(all_results)
    df = df.set_index("Model").sort_values("Accuracy", ascending=False)
    return df
