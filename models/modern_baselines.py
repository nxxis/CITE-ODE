"""Modern baseline models used for comparison: a small Transformer variant.

This module contains lightweight implementations of components used as
baselines in experiments (temporal Transformer). The implementations are
intentionally concise and documented for reproducibility.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer inputs.

    Args:
        d_model (int): embedding dimensionality
        max_len (int): maximum supported sequence length
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # register_buffer ensures the tensor moves with the module (cpu/cuda)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [B, L, D] -> add positional encodings on time dimension
        return x + self.pe[:, : x.size(1), :]


class TSTransformer(nn.Module):
    """A compact Transformer encoder used as a temporal baseline.

    The model projects input features to a `d_model` embedding, adds positional
    encodings, runs a Transformer encoder, and classifies using the last valid
    token in each sequence (according to the provided mask).
    """

    def __init__(
        self, input_dim=4, d_model=64, n_heads=4, num_layers=3, dropout=0.1
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, mask):
        """Forward pass.

        Args:
            x: [B, L, input_dim] input feature sequences (padded)
            mask: boolean mask [B, L] (True for valid tokens)

        Returns:
            logits: [B] classification logits for each sequence
        """

        x = self.proj(x)  # [B, L, d_model]
        x = self.pos_enc(x)
        # Transformer expects True for padding positions in `src_key_padding_mask`
        src_key_padding_mask = (mask == 0)
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Select the last valid token per example and classify
        idx_last = mask.sum(dim=1).long() - 1
        final = out[torch.arange(out.size(0)), idx_last]
        return self.classifier(final).squeeze(-1)
