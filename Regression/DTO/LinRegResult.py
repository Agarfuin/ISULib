from dataclasses import dataclass

@dataclass
class LinRegResult:
    slope: float
    intercept: float
    corr_coef: float
    r_squared: float
    equation: str
    
    def predict(self, x_val):
        return x_val*self.slope+self.intercept