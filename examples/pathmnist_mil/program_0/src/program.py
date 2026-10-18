"""Gated attention MIL baseline; frozen image features and bag labels only.

Pooling follows Ilse et al., ICML 2018. The feature extractor and small network
are a pilot implementation, not a reproduction of that paper's experiments.
"""
import torch
from torch import nn


class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.attention_v = nn.Linear(32, 16)
        self.attention_u = nn.Linear(32, 16)
        self.attention_w = nn.Linear(16, 1, bias=False)
        self.classifier = nn.Linear(32, 1)

    def forward(self, bags):
        features = self.encoder(bags)
        scores = self.attention_w(
            torch.tanh(self.attention_v(features)) * torch.sigmoid(self.attention_u(features))
        ).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * features).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


def build_model(input_dim):
    return GatedAttentionMIL(input_dim)

