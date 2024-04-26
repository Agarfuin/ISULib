import numpy as np
from util.util import util
from Regression.DTO.LinRegResult import LinRegResult

rng = np.random.default_rng(seed=2024)

class LinReg:
    @staticmethod
    def linear_regression(independent_vals, dependent_vals):
        essentials = LinReg.__calculate_essentials(independent_vals, dependent_vals)
        b1 = LinReg.__calculate_b1(essentials.get("numerator"), essentials.get("denominator"))
        b0 = LinReg.__calculate_b0(independent_vals.mean(), dependent_vals.mean(), b1)
        cc = LinReg.__calculate_cc(essentials.get("numerator"), essentials.get("ssd_x"), essentials.get("ssd_y"))
        r2 = LinReg.__calculate_r2(independent_vals, dependent_vals, b0, b1, essentials.get("ssd_y"))
        eq = f"f(x)={b1}x+{b0}"
        return LinRegResult(b1, b0, cc, r2, eq)
    
    @staticmethod
    def multiple_regression(train_x, train_y, vaild_x, valid_y, epoch=10, learning_rate=1e-4):
        losses = []
        params = LinReg.__init_params(train_x.shape[1])
        for i in range(epoch):
            prediction = LinReg.__forward_propagation(params, train_x)
            grad = util.mse_gradient(train_y, prediction)
            params = LinReg.__backward_propagation(params, train_x, learning_rate, grad)
            loss = util.mse(train_y, prediction)
            losses.append(loss)
            if i % 100 == 0:
                prediction = LinReg.__forward_propagation(params, vaild_x)
                valid_loss = util.mse(valid_y, prediction)
                print(f"Epoch {i}\nloss: {losses[i]} - val_loss: {valid_loss}")
        return LinRegResult(slopes=params[0], intercept=params[1], losses=losses)
    
    @staticmethod
    def __init_params(num_predictors):
        weights = (rng.random((num_predictors, 1)) * 2) - 1
        biases = np.ones((1,1))
        return [weights, biases]
    
    @staticmethod
    def __forward_propagation(params, x):
        weights, biases = params
        prediction = x @ weights + biases
        return prediction
    
    @staticmethod
    def __backward_propagation(params, x, learning_rate, gradient):
        w_gradient = (x.T / x.shape[0]) @ gradient
        b_gradient = np.mean(gradient, axis=0)
        
        params[0] -= w_gradient * learning_rate
        params[1] -= b_gradient * learning_rate
        
        return params
    
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
    def __calculate_r2(x_vals, y_vals, b0, b1, ssd_y):
        TSS = ssd_y
        SSE = 0
        for x_val, y_val in zip(x_vals, y_vals):
            SSE += (y_val-(b0+b1*x_val))**2
        return 1-(SSE/TSS)