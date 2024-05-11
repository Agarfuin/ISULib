from ANN.Neuron import Neuron
from ANN.Layer import Layer
from util.util import util
from typing import List
from ANN.DTO.ModelResult import ModelResult
import numpy as np

rng = np.random.default_rng(seed=2024)

class Model:
    def __init__(self):
        self.layers: List[Layer] = []
    
    def add_layer(self, num_of_neurons=0, activation_func='sigmoid'):
        layer = self.__create_layer(num_of_neurons, activation_func)
        self.layers.append(layer)
        print(f"Dense layer added with:\tNeurons: {len(layer.neurons)}\tActivation: {layer.activation_func}\tParam: {layer.param}")
        
    def fit(self, x_train, y_train, x_test, y_test, epochs, batch_size, learning_rate, x_valid=None, y_valid=None):
        for epoch in range(epochs):
            epoch_loss = []
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i:(i+batch_size)]
                y_batch = y_train[i:(i+batch_size)]
                batch = self.__forward_propagation(x_batch)
                hidden = self.__get_hidden()
                
                loss = util.mse_gradient(y_batch, batch)
                epoch_loss.append(np.mean(loss ** 2))
                
                self.__back_propagation(hidden, loss, learning_rate)
                
            accuracy = self.__calculate_accuracy(ModelResult(self.layers), x_test, y_test)
            
            if x_valid is not None and y_valid is not None:
                valid_preds = self.__forward_propagation(x_valid)
                val_loss = util.mse(valid_preds, y_valid)
                val_accuracy = self.__calculate_accuracy(ModelResult(self.layers), x_valid, y_valid)
                print(f"Epoch: {epoch} - Train MSE: {np.mean(epoch_loss):.4f} - Accuracy: {accuracy:.4f} - Valid MSE: {np.mean(val_loss):.4f} - Valid Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch: {epoch} - Train MSE: {np.mean(epoch_loss):.4f} - Accuracy: {accuracy:.4f}")
        return ModelResult(self.layers)
        
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
        
    def __calculate_accuracy(self, model, x_test, y_test):
        correct_guess = 0
        for (x, y) in zip(x_test, y_test):
            guess = np.argmax(model.predict(x))
            actual = np.argmax(y)
            if guess == actual:
                correct_guess+=1
        return correct_guess / len(x_test)
    
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
        self.layers[0].set_activations(batch)
        for index, layer in enumerate(self.layers[:-1]):
            layer_weights = layer.get_weights()
            layer_biases = self.layers[index+1].get_biases()
            batch = np.matmul(batch, layer_weights) + layer_biases
            if layer.activation_func == "sigmoid":
                batch = util.sigmoid(batch)
            elif layer.activation_func == "relu":
                batch = util.relu(batch)
            elif layer.activation_func == "softmax":
                batch = util.softmax(batch)
            else:
                raise TypeError(f"Activation function is not suitable.\nNeeded: sigmoid, relu, softmax\nGiven:{layer.activation_func}")
            self.layers[index+1].set_activations(batch)
        return batch
            
    def __back_propagation(self, hidden, grad, lr):
        for index in range(len(self.layers)-1, -1, -1):
            if index != len(self.layers) - 1:
                grad = np.multiply(grad, np.heaviside(hidden[index+1], 0))
            
            w_grad = hidden[index].T @ grad
            b_grad = np.mean(grad, axis=0)
            
            if index != len(self.layers) - 1:
                self.layers[index+1].set_biases(self.layers[index+1].get_biases()-(b_grad*lr))
                self.layers[index].set_weights(self.layers[index].get_weights()-(w_grad*lr))
                grad = grad @ self.layers[index].get_weights().T
                
    def __get_hidden(self):
        return [layer.get_activations() for layer in self.layers]