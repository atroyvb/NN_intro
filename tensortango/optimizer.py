"""How do we update parameters over time"""
from tensortango import mlp

class Optimizer():
    """ Goal is to make it more efficient, acting on the MLP, and make the learning rate faster. """
    def __init__(self, neural_network: mlp.MLP, learning_rate: float = 0.01):
        self.net = neural_network
        self.lr = learning_rate  # an exponential decay bc we're trying to minimize our error 

    def step(self):
        raise NotImplementedError
    


class SGD(Optimizer):
    def step(self):
        for param, grad in self.net.params_and_grads():
            param -= grad*self.lr 
