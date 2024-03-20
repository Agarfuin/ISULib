import numpy as np
import pandas as pd
from Regression.DTO.LinRegResult import LinRegResult

class LinReg:
    @staticmethod
    def linear_regression(independent_vals, dependent_vals):
        if LinReg.__is_multidimensional(independent_vals):
            x_means = []
            b_vals = np.dot(np.dot(np.linalg.inv(np.dot(independent_vals.T, independent_vals)), independent_vals.T), dependent_vals)
            for category in independent_vals:
                x_means.append(independent_vals[category].mean())
            b0 = LinReg.__calculate_b0(x_means, dependent_vals.mean(), b_vals)
            cc = LinReg.__calculate_cc_mult(independent_vals, dependent_vals, b_vals, b0)
            r2 = LinReg.__calculate_r2_mult(independent_vals, dependent_vals, b_vals, b0)
            formatted_xis = [f"x{i+1}" for i in range(len(b_vals))]
            eq = "f(" + ",".join(formatted_xis) + ")="
            for i in range(len(b_vals)):
                eq += f"{b_vals[i]}x{i+1}+"
            eq += f"{b0}"
            return LinRegResult(b_vals, b0, cc, r2, eq)
        else:
            essentials = LinReg.__calculate_essentials(independent_vals, dependent_vals)
            b1 = LinReg.__calculate_b1(essentials.get("numerator"), essentials.get("denominator"))
            b0 = LinReg.__calculate_b0(independent_vals.mean(), dependent_vals.mean(), b1)
            cc = LinReg.__calculate_cc(essentials.get("numerator"), essentials.get("ssd_x"), essentials.get("ssd_y"))
            r2 = LinReg.__calculate_r2(independent_vals, dependent_vals, b0, b1, essentials.get("ssd_y"))
            eq = f"f(x)={b1}x+{b0}"
            return LinRegResult(b1, b0, cc, r2, eq)
        
    @staticmethod
    def __calculate_essentials(x_vals, y_vals):
        sod_x, sod_y, ssd_x, ssd_y, numerator, denominator = 0, 0, 0, 0, 0, 0
        for x_val, y_val in zip(x_vals, y_vals):
            sod_x += x_val-x_vals.mean()
            sod_y += y_val-y_vals.mean()
            ssd_x += (x_val-x_vals.mean())**2
            ssd_y += (y_val-y_vals.mean())**2
            numerator += (x_val-x_vals.mean())*(y_val-y_vals.mean())
            denominator += (x_val-x_vals.mean())**2
        return_dict = {
            "sod_x": sod_x,
            "sod_y": sod_y,
            "ssd_x": ssd_x,
            "ssd_y": ssd_y,
            "numerator": numerator,
            "denominator": denominator
        }
        return return_dict
    
    @staticmethod
    def __calculate_b1(numerator, denominator):
        return numerator/denominator
        
    @staticmethod
    def __calculate_b0(x_means, y_mean, b_vals):
        return y_mean-np.dot(b_vals, x_means)
    
    @staticmethod
    def __calculate_cc(numerator, ssd_x, ssd_y):
        return (numerator)/((ssd_x*ssd_y)**0.5)
    
    @staticmethod
    def __calculate_cc_mult(independent_vals, y_vals, b_vals, b0):
        sop_y = 0
        ssp_y = 0
        ssd_y = 0
        predicts = []
        for row in independent_vals.values:
            predict = np.dot(b_vals, row)+b0
            predicts.append(predict)
        for y_val, predict in zip(y_vals, predicts):
            sop_y += (predict-y_vals.mean())*(y_val-y_vals.mean())
            ssp_y += (predict-y_vals.mean())**2
            ssd_y += (y_val-y_vals.mean())**2
        return sop_y/((ssp_y*ssd_y)**0.5)
    
    @staticmethod
    def __calculate_r2(x_vals, y_vals, b0, b1, ssd_y):
        TSS = ssd_y
        SSE = 0
        for x_val, y_val in zip(x_vals, y_vals):
            SSE += (y_val-(b0+b1*x_val))**2
        return 1-(SSE/TSS)
    
    @staticmethod
    def __calculate_r2_mult(independent_vals, y_vals, b_vals, b0):
        ssp_y = 0
        ssd_y = 0
        predicts = []
        for row in independent_vals.values:
            predict = np.dot(b_vals, row)+b0
            predicts.append(predict)
        for y_val, predict in zip(y_vals, predicts):
            ssp_y += (y_val-predict)**2
            ssd_y += (y_val-y_vals.mean())**2
        return 1-(ssp_y/ssd_y)
    
    @staticmethod
    def __is_multidimensional(independent_vals):
        if isinstance(independent_vals, list):
            if not independent_vals:
                return False
            return any(isinstance(val, list) for val in independent_vals)
        elif isinstance(independent_vals, pd.DataFrame):
            for col in independent_vals.columns:
                if isinstance(independent_vals[col], pd.Series):
                    return True
            return False
        else:
            return False