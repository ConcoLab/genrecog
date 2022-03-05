"""
Preprocessor code
Authors
 * Amirali Ashraf 2022
"""
import numpy as np
import torch
import sklearn
from sklearn.utils import shuffle


class Preprocessor:
    def __init__(
            self,
            path,
    ):
        self.path = path
        # self.X = None
        # self.y = None
        # with np.load(self.path) as data:
        #     self.X = data['X'].astype(np.float32)
        #     self.y = data['y'].astype(np.int64)

    def as_numpy(self):
        with np.load(self.path) as data:
            X = data['X'].astype(np.float32)
            y = data['y'].astype(np.int64)
        return X, y

    def as_torch(self):
        with np.load(self.path) as data:
            X = data['X'].astype(np.float32)
            y = data['y'].astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(y)

    def as_shuffled_numpy(self):
        with np.load(self.path) as data:
            X = data['X'].astype(np.float32)
            y = data['y'].astype(np.int64)
        return shuffle(X, y)

    def as_shuffled_torch(self):
        X, y = self.as_shuffled_numpy()
        return torch.from_numpy(X), torch.from_numpy(y)




