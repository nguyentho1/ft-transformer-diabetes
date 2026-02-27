"""
ft_transformer.py
-----------------
Feature Tokenizer + Transformer (FT-Transformer) for tabular binary
classification.

Architecture summary:
  • Categorical features  → learned Embedding → d_model-dim token
  • Numerical features    → shared Linear layer → d_model-dim token
  • Tokens are *summed* and passed through a Transformer Encoder stack
  • A final Linear + Sigmoid head produces P(diabetes = 1)

Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular
Data" (NeurIPS 2021).  https://arxiv.org/abs/2106.11959
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for binary classification on tabular data.

    Parameters
    ----------
    numerical_features : list[str]
        Names (or count) of continuous input features.
    categorical_cardinalities : list[int]
        Number of unique categories for each categorical feature.
        One embedding table is created per entry.
    d_model : int
        Transformer hidden / embedding dimension (default: 128).
    nhead : int
        Number of self-attention heads (default: 4).
    num_layers : int
        Number of TransformerEncoder layers (default: 2).
    dim_feedforward : int
        Inner dimension of the FFN inside each Transformer layer (default: 256).
    """

    def __init__(
        self,
        numerical_features: list,
        categorical_cardinalities: list[int],
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
    ):
        super().__init__()

        # One embedding table per categorical feature
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in categorical_cardinalities]
        )

        # Project all numerical features together into d_model space
        self.numerical_linear = nn.Linear(len(numerical_features), d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,  # (seq, batch, feature) convention
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Binary classification head
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, numerical_data: torch.Tensor, categorical_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        numerical_data : torch.Tensor  shape (batch, n_numeric)
        categorical_data : torch.Tensor  shape (batch, n_categorical)

        Returns
        -------
        torch.Tensor  shape (batch, 1)  — predicted probability of class 1
        """
        # Embed each categorical column and sum them
        cat_embeds = [self.embeddings[i](categorical_data[:, i]) for i in range(len(self.embeddings))]
        categorical_embeds = torch.stack(cat_embeds, dim=1).sum(dim=1)  # (batch, d_model)

        # Encode numerical block
        numerical_embeds = self.numerical_linear(numerical_data)  # (batch, d_model)

        # Add-fuse feature tokens → shape (1, batch, d_model) for TransformerEncoder
        combined = (numerical_embeds + categorical_embeds).unsqueeze(0)

        # Transformer encoding
        transformer_out = self.transformer_encoder(combined)  # (1, batch, d_model)
        transformer_out = transformer_out.squeeze(0)          # (batch, d_model)

        # Binary output with sigmoid
        return torch.sigmoid(self.output_layer(transformer_out))


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def build_dataloaders(
    X_train,
    X_test,
    y_train,
    y_test,
    numeric_features: list,
    categorical_features: list,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    """
    Wrap train/test splits into PyTorch DataLoaders.

    Returns
    -------
    train_loader, test_loader
    """
    def _make_loader(X, y, shuffle: bool) -> DataLoader:
        num_tensor = torch.tensor(X[numeric_features].values.astype("float32"))
        cat_tensor = torch.tensor(X[categorical_features].values.astype("int64"))
        lbl_tensor = torch.tensor(y.values.astype("float32"))
        dataset = TensorDataset(num_tensor, cat_tensor, lbl_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = _make_loader(X_train, y_train, shuffle=True)
    test_loader = _make_loader(X_test, y_test, shuffle=False)
    return train_loader, test_loader
