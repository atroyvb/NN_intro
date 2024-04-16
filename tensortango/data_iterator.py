"""Create a mechanism for batching data for our neural network."""
from typing import Iterator 
import numpy as np 

from tensortango import tensor



class DataIterator():
    def __call__(self, features: tensor.Tensor, labels: tensor.Tensor) -> Iterator:
        """Batch a set of features and labels 

        Args:
            features (tensor.Tensor): features
            labels (tensor.Tensor): associated labels per row 

        Yields:
            Iterator: Subset of data to be trained upon
        """

        raise NotImplementedError
    

class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True):
        """Create a new iterator for batching data

        Args:
            batch_size (int, optional): number of values in a batch. Defaults to 32.
            shuffle (bool, optional): whether to shuffle training data. Defaults to True.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, features: tensor.Tensor, labels: tensor.Tensor) -> Iterator:
        starts = np.arange(0, len(features), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)
        
        for start in starts:
            end = start + self.batch_size
            batch_features = features[start:end]
            batch_labels = labels[start:end]
            yield (batch_features, batch_labels) # yield: returns 1 value, then everything happens, then returns 
                                                 # next value, and then everything happens etc. doesn't dump 
                                                 # everything into the memory

            

    


if __name__ == '__main__':
    pass 