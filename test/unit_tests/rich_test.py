import unittest
import torch
import numpy as np
from genrecog.preprocess.preprocessor import Preprocessor


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_preprocessor(self):
        preprocessor = Preprocessor('data/test.npz')
        assert preprocessor.X.shape == (10, 16_000)
        assert preprocessor.y.shape == (10,)
        assert isinstance(preprocessor.X, (np.ndarray, np.float32))
        assert isinstance(preprocessor.y, (np.ndarray, np.int64))


if __name__ == '__main__':
    unittest.main()
