"""Frozen host reference controls. Never import edited candidate code here."""
from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
from torch import nn

# The evaluator sets this to the frozen requested reference pooling.
POOLING = "hybrid"
_MODEL = None
_OPTIMIZER = None
_TRAIN_X = None
_TRAIN_Y = None
_CLIP = None
_SNAPSHOTS = {}


class MIL(nn.Module):
    def __init__(self, input_dim, pooling=POOLING):
        super().__init__()
        self.pooling = pooling
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.classifier = nn.Linear(32, 1)
        if pooling in {"attention", "hybrid"}:
            self.attention_v = nn.Linear(32, 16)
            self.attention_u = nn.Linear(32, 16)
            self.attention_w = nn.Linear(16, 1, bias=False)

    def forward(self, bags):
        h = self.encoder(bags)
        logits = self.classifier(h).squeeze(-1)
        if self.pooling == "mean":
            return logits.mean(1)
        if self.pooling == "max":
            return logits.max(1).values
        if self.pooling == "top2":
            return logits.topk(min(2, logits.shape[1]), dim=1).values.mean(1)
        lme = torch.logsumexp(logits, dim=1) - math.log(logits.shape[1])
        if self.pooling == "lme":
            return lme
        if self.pooling not in {"attention", "hybrid"}:
            raise ValueError("unknown pooling")
        attention = self.attention_w(
            torch.tanh(self.attention_v(h)) * torch.sigmoid(self.attention_u(h))
        ).squeeze(-1).softmax(1)
        score = (attention * logits).sum(1)
        return score if self.pooling == "attention" else 0.5 * score + 0.5 * lme


def handle(request):
    """reset(seed); init(train data, host optimizer); train(host batch IDs); predict(x).

    snapshot/restore use an opaque named checkpoint held inside the worker.
    All requests and replies are JSON values. predict returns raw logits.
    """
    global _MODEL, _OPTIMIZER, _TRAIN_X, _TRAIN_Y, _CLIP, _SNAPSHOTS
    op = request["op"]
    if op == "reset":
        seed = int(request["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        _MODEL = _OPTIMIZER = _TRAIN_X = _TRAIN_Y = _CLIP = None
        _SNAPSHOTS = {}
        return {"ready": True}
    if op == "init":
        _TRAIN_X = torch.tensor(request["train_x"], dtype=torch.float32)
        _TRAIN_Y = torch.tensor(request["train_y"], dtype=torch.float32)
        _MODEL = MIL(_TRAIN_X.shape[-1], POOLING).cpu()
        _OPTIMIZER = torch.optim.Adam(
            _MODEL.parameters(), lr=float(request["learning_rate"]),
            weight_decay=float(request["weight_decay"]),
        )
        _CLIP = float(request["gradient_clip_norm"])
        return {"parameters": sum(p.numel() for p in _MODEL.parameters())}
    if _MODEL is None:
        raise ValueError("reset and init are required")
    if op == "train":
        _MODEL.train()
        loss = None
        for indices in request["batch_indices"]:
            _OPTIMIZER.zero_grad(set_to_none=True)
            scores = _MODEL(_TRAIN_X[indices])
            loss = nn.functional.binary_cross_entropy_with_logits(scores, _TRAIN_Y[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(_MODEL.parameters(), _CLIP, error_if_nonfinite=True)
            _OPTIMIZER.step()
        return {"loss": float(loss.detach()), "steps": len(request["batch_indices"])}
    if op == "predict":
        if set(request) != {"op", "x"}:
            raise ValueError("predict accepts features only")
        _MODEL.eval()
        with torch.no_grad():
            result = _MODEL(torch.tensor(request["x"], dtype=torch.float32))
        return {"logits": result.tolist()}
    if op == "snapshot":
        _SNAPSHOTS[request["name"]] = copy.deepcopy(_MODEL.state_dict())
        return {"saved": True}
    if op == "restore":
        _MODEL.load_state_dict(_SNAPSHOTS[request["name"]])
        return {"restored": True}
    raise ValueError(f"unknown operation: {op}")
