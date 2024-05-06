from dataclasses import dataclass, field
from typing import List
from ANN.Layer import Layer
from util.util import util

@dataclass
class ModelResult:
    layers: List[Layer] = field(default=None)
    
    def predict(self, x_test):
        for index, layer in enumerate(self.layers[:-1]):
            layer_weights = layer.get_weights()
            layer_biases = self.layers[index+1].get_biases()
            x_test = (x_test @ layer_weights) + layer_biases
            if layer.activation_func == "sigmoid":
                x_test = util.sigmoid(x_test)
            elif layer.activation_func == "relu":
                x_test = util.relu(x_test)
            elif layer.activation_func == "softmax":
                x_test = util.softmax(x_test)
            else:
                raise TypeError(f"Activation function is not suitable.\nNeeded: sigmoid, relu, softmax\nGiven:{layer.activation_func}")
        return x_test