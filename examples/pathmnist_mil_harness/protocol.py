"""Frozen evaluator protocol. This module is outside editable candidate paths."""
SEARCH_SEEDS = (17, 29)
FINAL_SEEDS = (101, 211, 307, 401, 503)
NOISE_RATES = (0.0, 0.15)
BAG_SIZES = (16, 32, 64)
MAX_STEPS = 640
CHECK_EVERY = 160
BATCH_SIZE = 32
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP_NORM = 5.0
MAX_PARAMETERS = 20000
SMOKE_STEPS = 8
CONTROL_POOLINGS = ("attention", "lme", "mean", "max", "top2")
