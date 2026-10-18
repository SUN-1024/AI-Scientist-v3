"""Log-mean-exp MIL control using frozen image features and bag labels only."""
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
        instance_logits = self.classifier(features).squeeze(-1)
        instance_dim = 1
        temperature = 1.0
        # The interface supplies complete dense bags, with no padded instances.
        valid_count = instance_logits.new_tensor(instance_logits.shape[instance_dim])
        scaled_logits = instance_logits / temperature
        pooled_logits = temperature * (torch.logsumexp(scaled_logits, dim=instance_dim) - valid_count.log())
        return pooled_logits


def build_model(input_dim):
    return GatedAttentionMIL(input_dim)

