import numpy as np
import torch
from genrecog.preprocess.feature import Feature


def test_torch_fbank_features():
    feature_maker = Feature(sample_rate=16_000, n_fft=400)
    tensor = torch.rand([10, 16_000])
    features = feature_maker.torch_fbank_features(tensor)
    assert features.shape == (10, 101, 40)

def test_numpy_fbank_features():
    feature_maker = Feature()
    np_array = np.random.randn(10, 16_000)
    features = feature_maker.numpy_fbank_features(np_array)
    assert features.shape == (10, 101, 40)
