from typing import NamedTuple, Optional
from torch import Tensor


class Memory(NamedTuple):
    state: Tensor
    action: int
    next_state: Optional[Tensor]
    reward: float
    done: bool
