import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple
import matplotlib.pyplot as plt



def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    number_of_samples = train_x.shape[0]

    indices = np.arange(number_of_samples)
    np.random.shuffle(indices)

    for start_index in range(0, number_of_samples, batch_size):
        last_index = start_index + batch_size
        batch_indices=indices[start_index:last_index]
        yield train_x[batch_indices], train_y[batch_indices]



class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x*(1-x)
    


class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - x ** 2



class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0,x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        # Subtract max for numerical stability
        x_stable = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_stable)
        softmax = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        return softmax

    def derivative(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        s = self.forward(x, axis=axis)
        batch_size, n = x.shape
        jacobian = np.empty((batch_size, n, n))
        for i in range(batch_size):
            # Compute the Jacobian for each sample in the batch
            jacobian[i] = np.diag(s[i]) - np.outer(s[i], s[i])
        return jacobian

class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # If y_true is one-dimensional, calculate element-wise loss.
        if y_true.ndim == 1:
            return 0.5 * (y_pred - y_true) ** 2
        else:
            # For multi-dimensional outputs, sum the squared differences for each sample.
            return 0.5 * np.sum((y_pred - y_true) ** 2, axis=1)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true

class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-12, 1.0)
        # Compute loss for each sample; sum over classes (axis=1)
        return -np.sum(y_true * np.log(y_pred_clipped), axis=1)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true
    


class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        # this will store the activations (forward prop)
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None

        # calculating the glorot initialization
        limit = np.sqrt(6/(fan_in + fan_out))

        # Initialize weights and biaes
        self.W = np.random.uniform(-limit, limit, size=(fan_in, fan_out)) # weights
        self.b = np.zeros(fan_out)  # biases

    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        z = np.dot(h, self.W) + self.b
        self.activations = self.activation_function.forward(z)
        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        if isinstance(self.activation_function, Softmax):
            self.delta = delta
        else:
            activation_function_derivative = self.activation_function.derivative(self.activations)
            self.delta = delta * activation_function_derivative
        dL_dW = np.dot(h.T, self.delta)
        dL_db = np.sum(self.delta, axis=0)
        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output, list of inputs to each layer including the input data
        """
        cache = [x]
        output = x
        for layer in self.layers:
            output = layer.forward(output)
            cache.append(output)

        return output, cache

    def backward(self, loss_grad: np.ndarray, cache:list) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param cache: list of activations from the forward pass
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        
        dl_dw_all = []
        dl_db_all = []
        delta = loss_grad

        #now we loop backwards
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            # the input we feed into layer i
            h=cache[i]
            # calculate the weight and bias gradient
            dL_dW, dL_db = layer.backward(h, delta)
            # add the weight and bias gradients to the list
            dl_dw_all.append(dL_dW)
            dl_db_all.append(dL_db)
            # computing the delta for the previous layer, still confused about this ask 
            if i>0:
                delta = np.dot(layer.delta, layer.W.T)

        dl_dw_all.reverse()
        dl_db_all.reverse()

        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        training_losses = []
        validation_losses = []

       
        for epoch in range(epochs):
            training_loss_for_epoch = 0.0
            num_batches = 0

            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                y_pred, cache = self.forward(batch_x)
                batch_loss = loss_func.loss(batch_y, y_pred)
                loss_value = np.mean(batch_loss)
                training_loss_for_epoch += loss_value
                num_batches +=1
                loss_gradient = loss_func.derivative(batch_y, y_pred)
                dW_all, db_all = self.backward(loss_gradient, cache)

                for i, layer in enumerate(self.layers):
                    layer.W -= learning_rate * dW_all[i]
                    layer.b -= learning_rate * db_all[i]
            #why this should be average
            average_training_loss = training_loss_for_epoch/num_batches
            training_losses.append(average_training_loss)

            val_pred, _ = self.forward(val_x)
            val_loss_values = loss_func.loss(val_y, val_pred)
            avg_val_loss = np.mean(val_loss_values)
            validation_losses.append(avg_val_loss)

            print(f"Current epoch: {epoch+1}/{epochs}")
            print(f"Training Loss: {average_training_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")



        return np.array(training_losses), np.array(validation_losses)