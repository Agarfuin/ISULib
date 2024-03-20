import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class LinRegResult:
    slopes: List[float]
    intercept: float
    corr_coef: float
    r_squared: float
    equation: str
    
    def predict(self, x_vals):
        return np.dot(x_vals, self.slopes)+self.intercept