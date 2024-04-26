from ANN.Neuron import Neuron
from ANN.Layer import Layer
from util.util import util
from typing import List
import numpy as np

rng = np.random.default_rng(seed=2024)

class Model:
    def __init__(self):
        self.layers: List[Layer] = []
    
    def add_layer(self, num_of_neurons=0, activation='sigmoid'):
        layer = self.__create_layer(num_of_neurons, activation)
        self.layers.append(layer)
        print(f"Dense layer added with:\tNeurons: {len(layer.neurons)}\tActivation: {layer.activation_func}\tParam: {layer.param}")
        
    def fit(self, x_train, y_train, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            epoch_loss = []
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i:(i+batch_size)]
                y_batch = y_train[i:(i+batch_size)]
                batch, hidden = self.__forward_propagation(x_batch)
                
                loss = util.mse_gradient(batch, y_batch)
                epoch_loss.append(np.mean(loss ** 2))
                
                for _ in self.layers:
                    self.__back_propagation(hidden, loss, learning_rate)
        
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
        
    # TODO: COMPLETE BACK PROPAGATION
    def test_back_prop(self, hidden, grad, lr):
        for index in range(len(self.layers)-1, 0, -1):
            if index != len(self.layers) - 1:
                grad = np.multiply(grad, np.heaviside(hidden[index+1], 0))
                
            w_grad = hidden[index].T @ grad
            b_grad = np.mean(grad, axis=0)
            
            self.layers[index].set_weights(self.layers[index].get_weights()-(w_grad*lr))
            self.layers[index].set_biases(self.layers[index].get_biases()-(b_grad*lr))
            grad = grad @ self.layers[index].get_weights().T
            
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def __relu(self, x):
        return np.maximum(x, 0)
    
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
        
    def __forward_propagation(self, batch):
        if not isinstance(batch, np.ndarray):
            batch = np.array(batch)
        hidden = [batch.copy()]
        for layer in self.layers[1:]:
            layer_weights = layer.get_weights()
            layer_biases = layer.get_biases()
            batch = np.matmul(batch, layer_weights.T) + layer_biases
            if layer.activation_func == "sigmoid":
                batch = self.__sigmoid(batch)
            elif layer.activation_func == "relu":
                batch = self.__relu(batch)
            else:
                raise TypeError(f"Activation function is not suitable.\nNeeded: sigmoid, relu\nGiven:{layer.activation_func}")
            layer.set_activations(batch[0])
            hidden.append(batch.copy())
        return batch, hidden
            
    def __back_propagation(self, hidden, grad, lr):
        for index in range(len(self.layers)-1, -1, -1):
            if index != len(self.layers) - 1:
                grad = np.multiply(grad, np.heaviside(hidden[index+1], 0))
                
            w_grad = hidden[index].T @ grad
            b_grad = np.mean(grad, axis=0)
            
            self.layers[index].set_weights(self.layers[index].get_weights()-(w_grad*lr))
            self.layers[index].set_biases(self.layers[index].get_biases()-(b_grad*lr))
            print(grad.shape)
            print(self.layers[index].get_weights().shape)
            grad = grad @ self.layers[index].get_weights()