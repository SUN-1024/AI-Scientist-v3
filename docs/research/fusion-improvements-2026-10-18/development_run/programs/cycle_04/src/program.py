"""Gated attention MIL midpoint control; frozen features and bag labels only.

Gated scores follow Ilse et al., ICML 2018, with equal sparsemax and softmax
weights. This pilot control is not a reproduction of paper experiments.
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
        scores = self.attention_w(torch.tanh(self.attention_v(features)) * torch.sigmoid(self.attention_u(features))).squeeze(-1)
        centered_scores = scores - scores.amax(dim=1, keepdim=True)
        sorted_scores = torch.sort(centered_scores, dim=1, descending=True).values
        ranks = torch.arange(1, scores.size(1) + 1, device=scores.device, dtype=scores.dtype).unsqueeze(0)
        cumulative = sorted_scores.cumsum(dim=1)
        support_size = (1.0 + ranks * sorted_scores > cumulative).sum(dim=1, keepdim=True)
        threshold = (cumulative.gather(1, support_size - 1) - 1.0) / support_size.to(scores.dtype)
        sparse_weights = torch.clamp(centered_scores - threshold, min=0.0)
        dense_weights = torch.softmax(scores, dim=1)
        weights = 0.5 * sparse_weights + 0.5 * dense_weights
        pooled = (weights.unsqueeze(-1) * features).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


def build_model(input_dim):
    return GatedAttentionMIL(input_dim)

