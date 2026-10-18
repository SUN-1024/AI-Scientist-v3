"""Editable whole-idea seed. All evaluation labels remain in the host evaluator."""
from __future__ import annotations

import copy
import math
import os
import random

import numpy as np
import torch
from torch import nn

# This seed is a known attention/LME mixture, not a novelty claim. Change a complete
# method or its joint configuration. Keep the host budget and JSON interface intact.
POOLING = "hybrid"
CONFIGURATION = {
    "learning_rate": 0.003, "weight_decay": 0.0001, "batch_size": 32,
    "hidden_dim": 32, "dropout": 0.0, "pooling_temperature": 1.0,
    "gradient_clip_norm": 5.0, "hybrid_mix": 0.5,
}
_MODEL = _OPTIMIZER = _TRAIN_X = _TRAIN_Y = _CLIP = _DEVICE = None
_SNAPSHOTS = {}


def assigned_device():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one assigned CUDA device is required; CPU fallback is disabled")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    uuid = getattr(properties, "uuid", None)
    if uuid is None:
        raise ValueError("the contracted Torch runtime must expose the CUDA device UUID")
    return torch.device("cuda:0"), {
        "device_type": "cuda", "cuda_available": True, "device_count": torch.cuda.device_count(),
        "logical_device_index": torch.cuda.current_device(), "device_name": properties.name,
        "device_uuid": str(uuid), "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }


class MIL(nn.Module):
    def __init__(self, input_dim, config, pooling=POOLING):
        super().__init__()
        width = config["hidden_dim"]
        self.pooling = pooling
        self.temperature = config["pooling_temperature"]
        self.mix = config["hybrid_mix"]
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(config["dropout"]))
        self.classifier = nn.Linear(width, 1)
        if pooling in {"attention", "hybrid"}:
            self.attention_v = nn.Linear(width, 16)
            self.attention_u = nn.Linear(width, 16)
            self.attention_w = nn.Linear(16, 1, bias=False)

    def forward(self, bags):
        hidden = self.encoder(bags)
        logits = self.classifier(hidden).squeeze(-1)
        if self.pooling == "mean":
            return logits.mean(1)
        if self.pooling == "max":
            return logits.max(1).values
        if self.pooling == "top2":
            return logits.topk(min(2, logits.shape[1]), dim=1).values.mean(1)
        lme = self.temperature * (torch.logsumexp(logits / self.temperature, dim=1) - math.log(logits.shape[1]))
        if self.pooling == "lme":
            return lme
        if self.pooling not in {"attention", "hybrid"}:
            raise ValueError("unknown pooling")
        attention = self.attention_w(
            torch.tanh(self.attention_v(hidden)) * torch.sigmoid(self.attention_u(hidden))
        ).squeeze(-1).softmax(1)
        score = (attention * logits).sum(1)
        return score if self.pooling == "attention" else self.mix * score + (1 - self.mix) * lme


def handle(request):
    global _MODEL, _OPTIMIZER, _TRAIN_X, _TRAIN_Y, _CLIP, _DEVICE, _SNAPSHOTS
    op = request["op"]
    if op == "get_configuration":
        if set(request) != {"op"}:
            raise ValueError("get_configuration accepts no data")
        return {"configuration": copy.deepcopy(CONFIGURATION)}
    if op == "reset":
        seed = int(request["seed"])
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _MODEL = _OPTIMIZER = _TRAIN_X = _TRAIN_Y = _CLIP = _DEVICE = None
        _SNAPSHOTS = {}
        return {"ready": True}
    if op == "init":
        config = request["configuration"]
        if config != CONFIGURATION:
            raise ValueError("host-approved configuration differs from the frozen candidate declaration")
        _DEVICE, device_report = assigned_device()
        torch.cuda.reset_peak_memory_stats(_DEVICE)
        _TRAIN_X = torch.tensor(request["train_x"], dtype=torch.float32, device=_DEVICE)
        _TRAIN_Y = torch.tensor(request["train_y"], dtype=torch.float32, device=_DEVICE)
        _MODEL = MIL(_TRAIN_X.shape[-1], config, POOLING).to(_DEVICE)
        _OPTIMIZER = torch.optim.Adam(_MODEL.parameters(), lr=request["learning_rate"],
                                      weight_decay=request["weight_decay"])
        _CLIP = request["gradient_clip_norm"]
        return {"parameters": sum(parameter.numel() for parameter in _MODEL.parameters()),
                "trainable_parameters": sum(parameter.numel() for parameter in _MODEL.parameters()
                                            if parameter.requires_grad),
                "device": device_report}
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
        torch.cuda.synchronize(_DEVICE)
        return {"loss": float(loss.detach().cpu()), "steps": len(request["batch_indices"])}
    if op == "predict":
        if set(request) != {"op", "x"}:
            raise ValueError("predict accepts features only")
        _MODEL.eval()
        with torch.no_grad():
            logits = _MODEL(torch.tensor(request["x"], dtype=torch.float32, device=_DEVICE))
        return {"logits": logits.cpu().tolist()}
    if op == "snapshot":
        _SNAPSHOTS[request["name"]] = copy.deepcopy(_MODEL.state_dict())
        return {"saved": True}
    if op == "restore":
        _MODEL.load_state_dict(_SNAPSHOTS[request["name"]])
        return {"restored": True}
    raise ValueError(f"unknown operation: {op}")
