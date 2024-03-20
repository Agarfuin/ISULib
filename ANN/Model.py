from ANN.Neuron import Neuron
from ANN.Layer import Layer
from typing import List
import numpy as np

rng = np.random.default_rng(seed=None)

class Model:
    def __init__(self):
        self.layers: List[Layer] = []
    
    def add_layer(self, num_of_neurons=0, activation='sigmoid'):
        layer = self.__create_layer(num_of_neurons, activation)
        self.layers.append(layer)
        print(f"Dense layer added with:\tNeurons: {len(layer.neurons)}\tActivation: {layer.activation_func}\tParam: {layer.param}")
        
    def fit(self, inputs):
        self.__forward_propagation(inputs)
        for layer in self.layers:
            for neuron in layer.neurons:
                print(f"activation: {neuron.activation}")
        
    def summary(self):
        data = [["Layers", "Neurons", "Param"]]
        for i in range(len(self.layers)):
            data.append([str(f"d_{i}({self.layers[i].activation_func})"), str(len(self.layers[i].neurons)), str(self.layers[i].param)])
        column_widths = [max(len(str(row[i])) for row in data) for i in range(len(data[0]))]
        print("-".join("-" * width for width in column_widths))
        for idx, row in enumerate(data):
            if idx == 1:
                print("=".join("=" * width for width in column_widths))
            print(" ".join(f"{value:<{width}}" for value, width in zip(row, column_widths)))
        print("=".join("=" * width for width in column_widths))
        print(f"Total params: {sum(layer.param for layer in self.layers)}")
        print("-".join("-" * width for width in column_widths))
        
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def __relu(self, x):
        return x if x >= 0 else 0
    
    def __create_layer(self, num_of_neurons, activation):
        if num_of_neurons <= 0:
            raise TypeError("Number of neurons have to be greater than 0")
        if activation!='sigmoid' and activation!='relu':
            raise TypeError("Pick an appropriate activation function")
        
        neurons = []
        
        for _ in range(num_of_neurons):
            if not self.layers:
                weights = None
            else:
                num_weights = len(self.layers[-1].neurons)
                weights = rng.random(num_weights) - 0.5
                
            neurons.append(Neuron(weights=weights))

        if len(self.layers) < 1:
            return Layer(neurons, activation, 0)
        else:
            param = len(self.layers[-1].neurons)*len(neurons)+len(neurons)
            return Layer(neurons, activation, param)
        
    def __forward_propagation(self, inputs):
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                print(f"weight: {neuron.weights}, bias: {neuron.bias}")
                weighted_sum = np.dot(inputs, neuron.weights) + neuron.bias
                print(f"sum: {weighted_sum}")
                if layer.activation_func == "sigmoid":
                    output = self.__sigmoid(weighted_sum)
                elif layer.activation_func == "relu":
                    output = self.__relu(weighted_sum)
                else:
                    raise TypeError(f"Activation function is not suitable.\nNeeded: sigmoid, relu\nGiven:{layer.activation_func}")
                print(f"output: {output}")
                neuron.activation = output
            inputs = [neuron.activation for neuron in layer.neurons]