from preprocess.preprocessor import Preprocessor
from preprocess.feature import Feature
import torch
import numpy as np


preprocessor = Preprocessor('../../data/test.npz')
X, y = preprocessor.as_shuffled_torch()

feature = Feature()
features = feature.mfcc_maker(X)
print('features.shape', features.shape)
print('features', features[0].shape)
print('features X', features[0][0])
print('features Y', features[0][1].shape)


