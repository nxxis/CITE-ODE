import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TSTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=64, n_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, mask):
        x = self.proj(x)                     # [B, L, d_model]
        x = self.pos_enc(x)
        src_key_padding_mask = (mask == 0)   # True for padding
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # Take the last valid token
        idx_last = mask.sum(dim=1).long() - 1
        final = out[torch.arange(out.size(0)), idx_last]
        return self.classifier(final).squeeze(-1)
