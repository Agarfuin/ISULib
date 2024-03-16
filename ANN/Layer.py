from ANN.Neuron import Neuron
from dataclasses import dataclass, field
from typing import List

@dataclass
class Layer:
    neurons: List[Neuron]
    activation_func: str
    param: int = field(default=0)