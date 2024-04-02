from dataclasses import dataclass, field
from typing import List

@dataclass
class LinRegResult:
    slopes: List[float] = field(default=None)
    intercept: float = field(default=0)
    corr_coef: float = field(default=None)
    r_squared: float = field(default=None)
    equation: str = field(default="No equation")
    losses: List[float] = field(default=None)
    
    def predict(self, x_vals):
        return (x_vals @ self.slopes + self.intercept)[0]