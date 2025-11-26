from collections import namedtuple

import torch


# Hyperparameters from the Paper (Section 5)
class Config:
    ENV_NAME = "BreakoutNoFrameskip-v4"
    BATCH_SIZE = 32
    GAMMA = 0.99  # Discount factor
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_DECAY_FRAMES = 1000000
    MEMORY_SIZE = 100000
    LR = 0.00025
    TARGET_UPDATE_FREQ = 10000
    LEARNING_FREQ = 4
    VALIDATION_FREQ = 5000
    TOTAL_FRAMES = 10000000

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)

config = Config()
