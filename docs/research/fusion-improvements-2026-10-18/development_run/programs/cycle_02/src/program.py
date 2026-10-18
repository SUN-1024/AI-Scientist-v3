"""Capacity-matched gated attention with attention-weighted log-mean-exp.

This combines established pooling components as a development control using
frozen image features and bag labels only. No method novelty is claimed.
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
        raw_instance_logits = self.classifier(features).squeeze(-1)
        instance_logits = 2.0 * torch.tanh(raw_instance_logits / 2.0)
        return torch.logsumexp(scores + instance_logits, dim=1) - torch.logsumexp(scores, dim=1)


def build_model(input_dim):
    return GatedAttentionMIL(input_dim)

