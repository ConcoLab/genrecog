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
        """
        This module helps to extract fbank features from a .wav sample.
        :param sample_rate: int
            sample rate.
        :param n_fft: int
            n_fft value.
        """
        self.feature_maker = Fbank(sample_rate=sample_rate, n_fft=n_fft)

    def torch_fbank_features(self, X):
        """
            Gets the wav sample and extracts the feature.
        :param X: np.darray
            Audio sample.
        :return: torch.Tensor
            Extracted features.
        """
        return self.feature_maker(X)

    def numpy_fbank_features(self, X):
        """
            Gets the wav sample and extracts the feature
        :param X: np.darray
            Audio sample.
        :return: np.darray
            Extracted features.
        """
        X = torch.from_numpy(X)
        return self.feature_maker(X).cpu().detach().numpy()
