from Regression.DTO.LinRegResult import LinRegResult

class LinReg:
    @staticmethod
    def linear_regression(x_vals, y_vals):
        essentials = LinReg.__calculate_essentials(x_vals, y_vals)
        b1 = LinReg.__calculate_b1(essentials.get("numerator"), essentials.get("denominator"))
        b0 = LinReg.__calculate_b0(x_vals, y_vals, b1)
        cc = LinReg.__calculate_cc(essentials.get("numerator"), essentials.get("ssd_x"), essentials.get("ssd_y"))
        r2 = LinReg.__calculate_r2(x_vals, y_vals, b0, b1, essentials.get("ssd_y"))
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
    def __calculate_b0(x_vals, y_vals, b1):
        return y_vals.mean()-b1*x_vals.mean()
    
    @staticmethod
    def __calculate_cc(numerator, ssd_x, ssd_y):
        return (numerator)/((ssd_x*ssd_y)**0.5)
    
    @staticmethod
    def __calculate_r2(x_vals, y_vals, b0, b1, ssd_y):
        TSS = ssd_y
        SSE = 0
        for x_val, y_val in zip(x_vals, y_vals):
            SSE += (y_val-(b0+b1*x_val))**2
        return 1-(SSE/TSS)