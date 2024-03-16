from ANN.Neuron import Neuron
from ANN.Layer import Layer
from typing import List
import numpy as np

class Model:
    def __init__(self):
        self.layers: List[Layer] = []
    
    def add_layer(self, num_of_neurons=0, activation='sigmoid'):
        if num_of_neurons<=0:
            raise TypeError("Number of neurons have to be greater than 0")
        if activation!='sigmoid' and activation!='relu':
            raise TypeError("Pick an appropriate activation function")
        
        neurons = []
        
        for _ in range(num_of_neurons):
            neurons.append(Neuron())

        layer = Layer(neurons, activation)
        if len(self.layers) < 1:
            param = 0
        else:
            param = len(self.layers[len(self.layers)-1].neurons)*len(layer.neurons)+len(layer.neurons)
        layer.param = param
        self.layers.append(layer)
        print(f"Dense layer added with:\tNeurons: {len(layer.neurons)}\tActivation: {layer.activation_func}\tParam: {layer.param}")
        
    def fit(self, inputs):
        self.__forward_propagation(inputs)
        
    def summary(self):
        total_param = 0
        data = [["Layers", "Neurons", "Param"]]
        for i in range(len(self.layers)):
            data.append([str(f"d_{i}({self.layers[i].activation_func})"), str(len(self.layers[i].neurons)), str(self.layers[i].param)])
            total_param += self.layers[i].param
        column_widths = [max(len(str(row[i])) for row in data) for i in range(len(data[0]))]
        print("-".join("-" * width for width in column_widths))
        for idx, row in enumerate(data):
            if idx == 1:
                print("=".join("=" * width for width in column_widths))
            print(" ".join(f"{value:<{width}}" for value, width in zip(row, column_widths)))
        print("=".join("=" * width for width in column_widths))
        print(f"Total params: {total_param}")
        print("-".join("-" * width for width in column_widths))
        
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def __relu(self, x):
        return x if x >= 0 else 0

    def __forward_propagation(self, inputs):
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                weighted_sum = np.dot(inputs, neuron.weights) + neuron.bias
                if layer.activation_func == "sigmoid":
                    output = self.__sigmoid(weighted_sum)
                elif layer.activation_func == "relu":
                    output = self.__relu(weighted_sum)
                else:
                    raise TypeError(f"Activation function is not suitable.\nNeeded: sigmoid, relu\nGiven:{layer.activation_func}")
                neuron.activation = output
            inputs = [neuron.activation for neuron in layer.neurons]
                