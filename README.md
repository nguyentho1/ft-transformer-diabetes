# FT-Transformer for Diabetes Classification

A PyTorch implementation of the **Feature Tokenizer + Transformer (FT-Transformer)** for binary diabetes classification on the Frankfurt Diabetes dataset, with full comparison against five classical and modern baselines.

---

## Overview

This project applies the FT-Transformer architecture (Gorishniy et al., NeurIPS 2021) to tabular medical data and benchmarks it against Logistic Regression, Random Forest, XGBoost, LightGBM, and TabNet.

Key features:
- Mixed numerical + categorical feature tokenisation
- Early stopping on test AUC
- Training time and inference time reporting
- AUC, loss, and RMSE history plots
- SHAP and LIME explainability
- ROC curve comparison across all models

---

## Model Architecture

```
Numerical features  →  Linear(n_numeric, d_model)  ─┐
                                                      ├─ ADD → TransformerEncoder → Linear(d_model, 1) → Sigmoid
Categorical feature →  Embedding(cardinality, d_model)─┘
```

Default hyperparameters:

| Parameter        | Value |
|-----------------|-------|
| d_model          | 128   |
| nhead            | 4     |
| num_layers       | 2     |
| dim_feedforward  | 256   |
| learning rate    | 0.001 |
| batch size       | 128   |
| max epochs       | 800   |
| early stop patience | 10 |

---

## Dataset

**Frankfurt Diabetes Dataset** — a variant of the Pima Indians Diabetes dataset.

Features: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`  
Target: `Outcome` (0 = No Diabetes, 1 = Diabetes)

Place the raw CSV at:
```
data/raw/frankfurt_diabetes.csv
```

---

## Project Structure

```
ft_transformer_diabetes/
├── main.py                        # End-to-end pipeline entrypoint
├── requirements.txt
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Load, clean, impute, scale, split
│   ├── ft_transformer.py          # FTTransformer model + DataLoader helpers
│   ├── train.py                   # Training loop, early stopping, history
│   ├── baseline_models.py         # LR, RF, XGBoost, LightGBM, TabNet
│   └── evaluation.py              # Metrics, plots, ROC curves
│
├── notebooks/
│   └── FTTransformer_diabetes.ipynb   # Original exploratory notebook
│
├── data/
│   ├── raw/                       # ← place frankfurt_diabetes.csv here
│   └── processed/
│
└── outputs/                       # Generated figures, model weights, CSVs
```

---

## Quickstart

### 1. Clone

```bash
git  https://github.com/nguyentho1/ft-transformer-diabetes.git
cd ft-transformer-diabetes
```

### 2. Create environment

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add dataset

```bash
cp /path/to/frankfurt_diabetes.csv data/raw/frankfurt_diabetes.csv
```

### 5. Run pipeline

```bash
python main.py --data data/raw/frankfurt_diabetes.csv
```

Optional flags:
```bash
python main.py --epochs 800 --lr 0.001 --batch_size 128 --patience 10
```

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/ft_transformer_weights.pt` | Saved PyTorch model state dict |
| `outputs/loss_history.eps` | Train & test BCE loss curves |
| `outputs/auc_history.eps` | Per-epoch AUC curve |
| `outputs/rmse_history.eps` | Per-epoch RMSE curve |
| `outputs/ft_confusion_matrix.eps` | FT-Transformer confusion matrix |
| `outputs/roc_curves.eps` | All-model ROC comparison |
| `outputs/correlation_heatmap.eps` | Feature correlation heatmap |
| `outputs/model_comparison.csv` | Accuracy / Precision / Recall / F1 / Time table |

---

## Models Compared

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient-boosted trees |
| LightGBM | Fast gradient boosting |
| TabNet | Attentive tabular network |
| **FT-Transformer** | Feature Tokenizer + Transformer |

---

## Reference

> Gorishniy et al. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS 2021.  
> https://arxiv.org/abs/2106.11959

---

## License

MIT
