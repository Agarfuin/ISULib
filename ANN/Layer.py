from ANN.Neuron import Neuron
from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class Layer:
    neurons: List[Neuron]
    activation_func: str
    param: int = field(default=0)
    
    def get_weights(self):
        return np.array([neuron.weights for neuron in self.neurons])
    
    def set_weights(self, new_weights):
        for index, neuron in enumerate(self.neurons):
            neuron.weights = new_weights[index]
            
    def get_biases(self):
        return np.array([neuron.bias for neuron in self.neurons])
    
    def set_biases(self, new_biases):
        for index, neuron in enumerate(self.neurons):
            neuron.bias = new_biases[index]
            
    def get_activations(self):
        return np.array([[neuron.activation for neuron in self.neurons]])
    
    def set_activations(self, new_activations):
        for index, neuron in enumerate(self.neurons):
            neuron.activation = new_activations[index]