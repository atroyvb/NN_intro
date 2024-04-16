"""A layer is a single set of neurons and it has to be able to learn via backpropagation as well as 
run the netwrok forward/feedforward"""

import numpy as np 
from typing import Callable
from tensortango.tensor import tensor 

class Layer():
    def __init__(self):
        self.w = tensor.Tensor #weights: strength of a connection
        self.b = tensor.Tensor #bias: added to the weight 
        self.x = None #Can be thought of as inputs
        self.grad_w = 0
        self.grad_b = 0

    def forward(self, x : tensor.Tensor) -> tensor.Tensor:
        """Compute the forward pass of the neurons in a layer"""
        raise NotImplementedError
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """Compute the back propagation through the layer"""
        raise NotImplementedError
    
    
class Linear(Layer):
    def __init__(self, input_size : int, output_size: int):
        """Create a linear layer

        Args:
            input_size (int): the number of input values (batch_size, input_size)

            output_size (int): the number of output values to the next layer
            or the final size (batch_size, input_size)
        """
        
        super().__init__() #runs the initialization of layer
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size) #only one bias per layer

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """Compute y = w @ x + b 
        where @ is the matrix multiplication version of scalar * """

        self.x = x
        return self.x @ self.w + self.b

    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """We are going to have to compute the partial derivatives to figure out
        what the heck is going on.
        X = w*x + b
        y = f(x)
        dy/dw = f'(X) * x 
        dy/dx = f'(X) * w
        dy/db = f'(X)

        Now let's put this in tensor form (i.e. matrix math)
        dy/dx = f'(X) @ w.T
        dy/dw = x.T @ f'(X)
        dy/db = f'(X)
        """
        self.grad_b = np.sum(grad, axis = 0)
        self.grad_w = self.x.T @ grad
        return grad @ self.w.T
    


class Activation(Layer):
    def __init__(self, 
                 input_size:int, 
                 output_size:int,
                 f, 
                 f_prime):
        """Initialize an activation layer as generic layer that has a forward 
        function 

        Args:
            input_size (int): number of input values
            output_size (int): number of output values to the next layer
            or the final size
            f (_type_): f
            f_prime (_type_): derivative of f 
        """
        super().__init__()
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)
        self.f = f 
        self.f_prime = f_prime

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        self.x = x 
        return self.f(self.x @ self.w + self.b)
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        self.grad_b = np.sum(grad, axis = 0)
        self.grad_w = self.x.T @ grad
        grad = grad @ self.w.T
        return self.f_prime(self.x)*grad
    
def tanh(x: tensor.Tensor) -> tensor.Tensor:
    return np.tanh(x)

def tanh_prime(x: tensor.Tensor) -> tensor.Tensor:
    y = tanh(x)
    return 1 - y**2

def relu(x: tensor.Tensor) -> tensor.Tensor:
    return np.clip(x, 0)  #clipping at 0 -- not allowing it to go below 0 
    
def relu_prime(x: tensor.Tensor) -> tensor.Tensor:
    return (x > 0).astype(float) # makes a boolean of 0s to 1s and convert it to float 


class Tanh(Activation):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size, tanh, tanh_prime)

class relu(Activation):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size, relu, relu_prime)