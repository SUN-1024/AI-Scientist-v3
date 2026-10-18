"""Fixed screening budget; candidate configuration cannot change this protocol."""

SEARCH_SEEDS = (17, 29)
FINAL_SEEDS = (101, 211, 307, 401, 503)
NOISE_RATES = (0.0, 0.15)
BAG_SIZES = (16, 32, 64)
MAX_STEPS = 160
CHECK_EVERY = 40
SMOKE_STEPS = 8
MAX_PARAMETERS = 20000
CONTROL_POOLINGS = ("attention", "lme", "mean", "max", "top2")
DEFAULT_CONFIGURATION = {
    "learning_rate": 0.003,
    "weight_decay": 0.0001,
    "batch_size": 32,
    "hidden_dim": 32,
    "dropout": 0.0,
    "pooling_temperature": 1.0,
    "gradient_clip_norm": 5.0,
    "hybrid_mix": 0.5,
}
