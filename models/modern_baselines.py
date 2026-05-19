import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
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
    def __init__(self, input_dim=4, d_model=64, n_heads=4, num_layers=3):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, mask=None):
        x = self.pe(self.proj(x))
        pad_mask = (mask == 0) if mask is not None else None
        out = self.transformer(x, src_key_padding_mask=pad_mask)
        idx_last = torch.clamp(mask.sum(1).long() - 1, min=0) if mask is not None else -1
        return self.classifier(out[torch.arange(out.size(0)), idx_last])

try:
    from mamba_ssm import Mamba
    class MambaICUModel(nn.Module):
        def __init__(self, input_dim=4, d_model=64, num_layers=3):
            super().__init__()
            self.proj = nn.Linear(input_dim, d_model)
            self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(num_layers)])
            self.norm = nn.LayerNorm(d_model)
            self.classifier = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x, mask=None):
            h = self.proj(x)
            for layer in self.layers:
                h = layer(h)
            h = self.norm(h)
            idx_last = torch.clamp(mask.sum(1).long() - 1, min=0) if mask is not None else -1
            return self.classifier(h[torch.arange(h.size(0)), idx_last])
except ImportError:
    print("Mamba-SSM not installed. MambaICUModel will be unavailable.")
