"""
Feature code
Authors
 * Amirali Ashraf 2022
"""
import numpy as np
import torch
from speechbrain.lobes.features import Fbank


class Feature:
    def __init__(
            self,
            sample_rate=22050,
            n_fft=552
    ):
        self.feature_maker = Fbank(sample_rate=sample_rate, n_fft=n_fft)

    def torch_fbank_features(self, X):
        return self.feature_maker(X)

    def numpy_fbank_features(self, X):
        X = torch.from_numpy(X)
        return self.feature_maker(X).cpu().detach().numpy

