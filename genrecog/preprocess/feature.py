"""
Feature code
Authors
 * Amirali Ashraf 2022
"""
import numpy as np
import torch
from speechbrain.lobes.features import Fbank, MFCC


class Feature:
    def __init__(
            self,
            sample_rate=22050,
            n_fft=552
    ):
        self.feature_maker = Fbank(sample_rate=sample_rate, n_fft=n_fft)
        self.mfcc_maker = MFCC()

    def torch_fbank_features(self, X):
        return self.feature_maker(X)

    def numpy_fbank_features(self, X):
        X = torch.from_numpy(X)
        return self.feature_maker(X).cpu().detach().numpy()

    def torch_mfcc_features(self, x):
        return self.mfcc_maker(x)

    def numpy_mfcc_features(self, x):
        x = torch.from_numpy(x)
        return self.mfcc_maker(x).cpu().detach().numpy
