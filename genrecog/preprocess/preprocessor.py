"""
Preprocessor code
Authors
 * Amirali Ashraf 2022
"""
import torch
from sklearn.utils import shuffle
import numpy as np
from scipy.io.wavfile import read, write
from os import walk


class Convertor:
    def __init__(self, path):
        """
        This class converts all the sound files for genres to numpy array and stores
        them as test.npz and train.npz. Since each genre has 100 files, it chooses
        90 first files as train dataset and the remaining 10 as test dataset.
        :param path: str
            path to genres folder.

        Example
        -------------
        >>> convertor = Convertor('drive/MyDrive/genres/genres/')
        >>> convertor.convert()
        """
        self.path = path
        self.seven_seconds_bits = 22050 * 7

    def convert(self):
        """
        This method converts the .wav files as is mentioned in the class description.

        Example
        -------------
        >>> Convertor('path').convert()
        """
        X_train_all = np.array([], dtype=np.float32).reshape(0, self.seven_seconds_bits)
        X_test_all = np.array([], dtype=np.float32).reshape(0, self.seven_seconds_bits)
        y_train_all = np.array([], dtype=np.int64)
        y_test_all = np.array([], dtype=np.int64)
        for (dirpath, dirnames, filenames) in walk(self.path):
            genres = dirnames
            break
        print(genres)
        # gets all genres
        for index, genre in enumerate(genres):
            # walk through each folder
            for (dirpath, dirnames, filenames) in walk(self.path + genre):
                X_train = np.array([], dtype=np.float32).reshape(0, self.seven_seconds_bits)
                # takes first 90 songs
                for filename in filenames[0:-10]:
                    # Fs or bit-rate for all files is 22050
                    Fs, data = read(self.path + genre + '/' + filename)
                    np_data = np.array(data[:4 * self.seven_seconds_bits], dtype=np.float32)
                    np_data = np.expand_dims(np_data, axis=0)
                    np_data = np.reshape(np_data, (4, self.seven_seconds_bits))
                    X_train = np.concatenate((X_train, np_data), axis=0)
                y_train = np.full(X_train.shape[0], index)
                X_train_all = np.concatenate((X_train_all, X_train), axis=0)
                y_train_all = np.concatenate((y_train_all, y_train))
                X_test = np.array([], dtype=np.float32).reshape(0, self.seven_seconds_bits)
                # takes last 10 songs
                for filename in filenames[-10:]:
                    # Fs or bit-rate for all files is 22050
                    Fs, data = read(self.path + genre + '/' + filename)
                    np_data = np.array(data[:4 * self.seven_seconds_bits], dtype=np.float32)
                    np_data = np.expand_dims(np_data, axis=0)
                    np_data = np.reshape(np_data, (4, self.seven_seconds_bits))
                    X_test = np.concatenate((X_test, np_data), axis=0)
                y_test = np.full(X_test.shape[0], index)

                X_test_all = np.concatenate((X_test_all, X_test), axis=0)
                y_test_all = np.concatenate((y_test_all, y_test))
        # save the resulted test and train set to the files.
        np.savez(f'train', X=X_train_all, y=y_train_all)
        np.savez(f'test', X=X_test_all, y=y_test_all)


class Preprocessor:
    def __init__(
            self,
            path,
    ):
        """
            Gets the npz file location which is created by the convertor and
            allows generating numpy or torch tensor for further processing.
        :param path: str
            Path to the npz file
        """
        self.path = path

    def as_numpy(self):
        """
            loads the data which are defined in the path of the class
        :return: np.ndarray, np.ndarray
            features and classes
        """
        with np.load(self.path) as data:
            X = data['X'].astype(np.float32)
            y = data['y'].astype(np.int64)
        return X, y

    def as_torch(self):
        """
            loads the data which are defined in the path of the class
        :return: np.Tensor, np.Tensor
            features and classes
        """
        with np.load(self.path) as data:
            X = data['X'].astype(np.float32)
            y = data['y'].astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(y)

    def as_shuffled_numpy(self):
        """
            loads the data which are defined in the path of the class
            and shuffles them
        :return: np.ndarray, np.ndarray
            features and classes
        """
        with np.load(self.path) as data:
            X = data['X'].astype(np.float32)
            y = data['y'].astype(np.int64)
        return shuffle(X, y)

    def as_shuffled_torch(self):
        """
            loads the data which are defined in the path of the class
            and shuffles them
        :return: np.Tensor, np.Tensor
            features and classes
        """
        X, y = self.as_shuffled_numpy()
        return torch.from_numpy(X), torch.from_numpy(y)
