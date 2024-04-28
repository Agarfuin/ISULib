from ANN.Neuron import Neuron
from ANN.Layer import Layer
from util.util import util
from typing import List
import numpy as np

rng = np.random.default_rng(seed=2024)

class Model:
    def __init__(self):
        self.layers: List[Layer] = []
    
    def add_layer(self, num_of_neurons=0, activation_func='sigmoid'):
        layer = self.__create_layer(num_of_neurons, activation_func)
        self.layers.append(layer)
        print(f"Dense layer added with:\tNeurons: {len(layer.neurons)}\tActivation: {layer.activation_func}\tParam: {layer.param}")
        
    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, x_valid=None, y_valid=None):
        for epoch in range(epochs):
            epoch_loss = []
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i:(i+batch_size)]
                y_batch = y_train[i:(i+batch_size)]
                batch, hidden = self.__forward_propagation(x_batch)
                
                loss = util.mse_gradient(y_batch, batch)
                epoch_loss.append(np.mean(loss ** 2))
                
                self.__back_propagation(hidden, loss, learning_rate)
            
            if x_valid is not None and y_valid is not None:
                valid_preds, _ = self.__forward_propagation(x_valid)
                print(f"Epoch: {epoch} Train MSE: {np.mean(epoch_loss)} Valid MSE: {np.mean(util.mse(valid_preds, y_valid))}")
            else:
                print(f"Epoch: {epoch} Train MSE: {np.mean(epoch_loss)}")
        
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
        return np.maximum(x, 0)
    
    def __softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0)
    
    def __create_layer(self, num_of_neurons, activation_func):
        if num_of_neurons <= 0:
            raise TypeError("Number of neurons have to be greater than 0")
        if activation_func!='sigmoid' and activation_func!='relu' and activation_func!='softmax':
            raise TypeError("Pick an appropriate activation function")

        neurons = []
        for _ in range(num_of_neurons):
            weights = None
            neurons.append(Neuron(weights=weights))
            
        if len(self.layers) < 1:
            return Layer(neurons, activation_func, 0)
        else:
            for neuron in self.layers[-1].neurons:
                neuron.weights = rng.random(num_of_neurons) - 0.5
            param = len(self.layers[-1].neurons)*num_of_neurons+num_of_neurons
            return Layer(neurons, activation_func, param)
        
    def __forward_propagation(self, batch):
        if not isinstance(batch, np.ndarray):
            batch = np.array(batch)
        hidden = [batch.copy()]
        for index, layer in enumerate(self.layers[:-1]):
            layer_weights = layer.get_weights()
            layer_biases = self.layers[index+1].get_biases()
            batch = np.matmul(batch, layer_weights) + layer_biases
            if layer.activation_func == "sigmoid":
                batch = self.__sigmoid(batch)
            elif layer.activation_func == "relu":
                batch = self.__relu(batch)
            elif layer.activation_func == "softmax":
                batch = self.__softmax(batch)
            else:
                raise TypeError(f"Activation function is not suitable.\nNeeded: sigmoid, relu, softmax\nGiven:{layer.activation_func}")
            self.layers[index+1].set_activations(batch[0])
            hidden.append(batch.copy())
        return batch, hidden
            
    def __back_propagation(self, hidden, grad, lr):
        for index in range(len(self.layers)-1, -1, -1):
            if index != len(self.layers) - 1:
                grad = np.multiply(grad, np.heaviside(hidden[index+1], 0))
            
            w_grad = hidden[index].T @ grad
            b_grad = np.mean(grad, axis=0)
            
            if index != 0:
                self.layers[index].set_biases(self.layers[index].get_biases()-(b_grad*lr))
            if index != len(self.layers) - 1:
                self.layers[index].set_weights(self.layers[index].get_weights()-(w_grad*lr))
                grad = grad @ self.layers[index].get_weights().T