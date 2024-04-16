""""Create the framework for training our neural network"""

from tensortango import loss, mlp, optimizer, tensor, data_iterator
# mlp stands for multi-layer perceptron 

def train(neural_net: mlp.MLP, 
          features: tensor.Tensor, 
          labels: tensor.Tensor, 
          epochs: int = 5000, # one complete cycle of the forward and backward pass
          iterator = data_iterator.BatchIterator(), 
          loss_fn = loss.MSE(), 
          optimizer_obj = optimizer.SGD(), 
          learning_rate: float = 0.05):# usually between 1^-3 and 1^-5
    """Train a neural network, otherwise known as a multilayer perceptron or fully 
    connected feedforward network.

    Args:
        neural_net (mlp.MLP): a defined neural network
        features (tensor.Tensor): _description_
        labels (tensor.Tensor): _description_
        epochs (int, optional): numbers of rounds of forward/backward training. Defaults to 5000.
        iterator(_type_, optional): batch iterator. defaults to data_iterator.BatchIterator
        loss_fn (_type_, optional): loss function . Defaults to loss.MSE.
        optimizer_obj (_type_, optional): the mechanism that updates the learning rate.
            Defaults to optimizer.SGD.
        learning_rate (float, optional): the amount of error to include in each 
            backprop step . Defaults to 0.05.
    """
    optim = optimizer(neural_net, learning_rate)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in iterator(features, labels):
            features, labels = batch 
            predictions = neural_net.forward(features)  # the first result is features 
            epoch_loss =+ loss.loss(predictions, labels)  #batch[1] is labels 
            grad = loss.grad(predictions, labels)  #computing the error 
            neural_net.backward(grad)
            optim.step() #updating weights and biases 
            neural_net.zero_parameters()
        print(f'Epoch {epoch} has loss {epoch_loss}')


if __name__ == '__main__':
    import numpy as np 
    from tensortango import layer 

    # use XOR because linear functions cannot represent XOR
    features = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])

    # labels are going to be values of true and values of false 
    labels = np.array([
        [1,0], # false is 1, true is 0 
        [0,1], # T 
        [1,0], # T
        [1,1]  # F 
    ])

    neural_net = mlp.MLP([layer.Tanh(2,2)],
                         layer.Tanh(2,2))
    
    train(neural_net, features, labels)
    print(neural_net.forward(features))  # forward is the same thing as predict 


