"""
train.py
--------
Training loop for the FT-Transformer model with:
  • Binary Cross-Entropy loss
  • Adam optimiser (lr = 0.001)
  • Early stopping based on test AUC
  • Training-time and test-time measurement
  • History tracking: train loss, test loss, AUC, RMSE
"""

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

from src.ft_transformer import FTTransformer, build_dataloaders
from src.data_preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    lr=0.001,
    batch_size=128,
    num_epochs=800,
    early_stopping_patience=10,
)


def build_model(categorical_cardinalities: list[int], config: dict = None) -> FTTransformer:
    """Instantiate an FTTransformer with the given configuration."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    return FTTransformer(
        numerical_features=NUMERIC_FEATURES,
        categorical_cardinalities=categorical_cardinalities,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
    )


def train_ft_transformer(
    model: FTTransformer,
    train_loader,
    test_loader,
    X_test,
    y_test,
    config: dict = None,
) -> dict:
    """
    Train the FT-Transformer with early stopping on test AUC.

    Parameters
    ----------
    model : FTTransformer
    train_loader, test_loader : DataLoader
    X_test : pd.DataFrame
    y_test : pd.Series
    config : dict, optional
        Override DEFAULT_CONFIG keys.

    Returns
    -------
    dict with keys:
        model, train_loss_history, test_loss_history,
        auc_history, rmse_history, training_time, y_pred, y_pred_proba
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Prepare fixed test tensors for per-epoch evaluation
    num_test = torch.tensor(X_test[NUMERIC_FEATURES].values.astype("float32"))
    cat_test = torch.tensor(X_test[CATEGORICAL_FEATURES].values.astype("int64"))

    train_loss_history, test_loss_history, auc_history, rmse_history = [], [], [], []

    best_val_auc = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(cfg["num_epochs"]):
        # ---- Training ----
        model.train()
        train_loss_epoch = 0.0
        for num_batch, cat_batch, lbl_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(num_batch, cat_batch).squeeze()
            loss = criterion(outputs, lbl_batch)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        train_loss_history.append(train_loss_epoch / len(train_loader))

        # ---- Evaluation ----
        model.eval()
        test_loss_epoch = 0.0
        all_outputs, all_labels = [], []

        with torch.no_grad():
            for num_batch, cat_batch, lbl_batch in test_loader:
                outputs = model(num_batch, cat_batch).squeeze()
                loss = criterion(outputs, lbl_batch)
                test_loss_epoch += loss.item()
                all_outputs.extend(outputs.tolist())
                all_labels.extend(lbl_batch.tolist())

        test_loss_history.append(test_loss_epoch / len(test_loader))

        # AUC
        auc = roc_auc_score(all_labels, all_outputs)
        auc_history.append(auc)

        # RMSE
        rmse = float(np.sqrt(np.mean((np.array(all_labels) - np.array(all_outputs)) ** 2)))
        rmse_history.append(rmse)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{cfg['num_epochs']}]  "
                f"Train Loss: {train_loss_history[-1]:.4f}  "
                f"Test Loss: {test_loss_history[-1]:.4f}  "
                f"AUC: {auc:.4f}  RMSE: {rmse:.4f}"
            )

        # Early stopping
        if auc > best_val_auc:
            best_val_auc = auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= cfg["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch + 1}. Best AUC: {best_val_auc:.4f}")
            break

    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f}s")

    # Final predictions
    model.eval()
    with torch.no_grad():
        test_start = time.time()
        proba = model(num_test, cat_test).squeeze().numpy()
        test_time = time.time() - test_start

    y_pred = (proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Inference time: {test_time:.4f}s")

    return dict(
        model=model,
        train_loss_history=train_loss_history,
        test_loss_history=test_loss_history,
        auc_history=auc_history,
        rmse_history=rmse_history,
        training_time=training_time,
        test_time=test_time,
        y_pred=y_pred,
        y_pred_proba=proba,
    )
