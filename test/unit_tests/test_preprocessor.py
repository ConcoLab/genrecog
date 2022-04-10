import os
import torch
import numpy as np
from genrecog.preprocess.preprocessor import Preprocessor


def before_test(tmpdir):
    X = np.random.randn(10, 16_000)
    y = np.random.randn(10)
    np_file = os.path.join(tmpdir, "random.npz")
    np.savez(np_file, X=X, y=y)
    return np_file


def test_preprocessor(tmpdir):
    np_file = before_test(tmpdir)
    preprocessor = Preprocessor(np_file)
    assert preprocessor.path == os.path.join(tmpdir, "random.npz")


def test_as_torch(tmpdir):
    np_file = before_test(tmpdir)
    preprocessor = Preprocessor(np_file)
    X_torch, y_torch = preprocessor.as_torch()
    assert torch.is_tensor(X_torch)
    assert torch.is_tensor(y_torch)
    assert X_torch.shape == (10, 16_000)
    assert y_torch.shape == (10,)


def test_as_numpy(tmpdir):
    np_file = before_test(tmpdir)
    X_loaded, y_loaded = Preprocessor(np_file).as_numpy()
    assert X_loaded.shape == (10, 16_000)
    assert y_loaded.shape == (10,)
    assert isinstance(X_loaded, (np.ndarray, np.float32))
    assert isinstance(y_loaded, (np.ndarray, np.int64))


def test_as_shuffled_numpy(tmpdir):
    np_file = before_test(tmpdir)
    preprocessor = Preprocessor(np_file)
    X, y = preprocessor.as_shuffled_numpy()
    assert (X.shape == (10, 16_000))
    assert (y.shape == (10,))
    assert isinstance(X, (np.ndarray, np.float32))
    assert isinstance(y, (np.ndarray, np.int64))


def test_as_shuffled_torch(tmpdir):
    np_file = before_test(tmpdir)
    preprocessor = Preprocessor(np_file)
    X, y = preprocessor.as_shuffled_torch()
    assert (X.shape == (10, 16_000))
    assert (y.shape == (10,))
    assert torch.is_tensor(X)
    assert torch.is_tensor(y)
