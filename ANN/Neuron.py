import numpy as np
from dataclasses import dataclass, field
from typing import List

rng = np.random.default_rng(2024)

@dataclass
class Neuron:
    activation: float = field(default=0)
    weights: List[float] = field(default=None)
    bias: float = field(default=rng.random())