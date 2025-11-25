from collections import namedtuple

import torch


# Hyperparameters from the Paper (Section 5)
class Config:
    ENV_NAME = "PongNoFrameskip-v4"
    BATCH_SIZE = 32  # Source [182]
    GAMMA = 0.99  # Discount factor
    EPS_START = 1.0  # Source [182]
    EPS_END = 0.1  # Source [182]
    EPS_DECAY_FRAMES = 1000000  # Source [182]
    MEMORY_SIZE = 100000  # Source [183]
    LR = 0.00025  # Standard RMSProp learning rate
    TARGET_UPDATE_FREQ = 10000  # How often to update the fixed target parameters
    LEARNING_FREQ = 4  # How often to perform gradient updates (every N steps)
    VALIDATION_FREQ = 5000  # How often to compute validation metrics
    TOTAL_FRAMES = 10000000  # Source [183]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)

config = Config()
