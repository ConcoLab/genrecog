"""Library to train models and generate specific reports
Authors
 * Amirali Ashraf 2022
 * Richard Grand'Maison 2022
 * Hassan Torkaman 2022
"""

from genrecog.preprocess.feature import Feature
import torch
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sn
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix


class FbankTrainer():
    """
    This class is a parent class for FbanckTrainers which has children which work
    with NN models.
    """

    def __init__(self, model, optimizer, loss, train_dataloader, validation_dataloader, num_epochs):
        """
        This method initialize the model and other arguments mentioned here to run the trainer and
        also keeps track of information that we will use to generate reports.
        :param model: torch.nn.Model
            the model that needs to be trained.
        :param optimizer: torch.optim
            the optimizer that the model is trained with
        :param loss: torch.nn.CrossEntropyLoss
            the loss function that calculates the loss for the model.
        :param train_dataloader: torch.utils.data.DataLoader
            a torch DataLoader object which trains the model by iterating over it
        :param validation_dataloader: torch.utils.data.DataLoader
            a torch DataLoader which validates the model at each iteration
        :param num_epochs: int
            number of epochs
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.loss = loss
        self.train_losses = []
        self.validation_losses = []
        self.train_accuracies = []
        self.validation_accuracies = []
        self.feature_maker = Feature()
        self.validation_dataloader = validation_dataloader

    def accuracy(self, y_true, y_pred):
        """
        Returns the accuracy of the model
        :param y_true: list
            List of true values
        :param y_pred: list
            List of predicted values
        :return: float
            the accuracy

        Example
        ------------
        >>> y_true = [1,2,1]
        >>> y_pred = [1,0,1]
        >>> accuracy(y_true, y_pred)
        """
        return (torch.sum(y_true == y_pred) / y_pred.shape[0])

    def plot_loss(self, title):
        """
        plots the training loss vs the validation loss
        :param title: str
            the title of the chart.
        """
        plt.plot(self.train_losses)
        plt.plot(self.validation_losses)
        plt.legend(['Training loss', 'Validation loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(title)
        plt.savefig(f"images/{title.replace(' ', '_')}")
        plt.show()

    def plot_accuracies(self, title):
        """
        plots the accuracies of iterations
        :param title: str
            chart's title
        """
        plt.plot(self.train_accuracies)
        plt.plot(self.validation_accuracies)
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title(title)
        plt.savefig(f"images/{title.replace(' ', '_')}")
        plt.show()

    def plot_confusion_matrix(self, eval_loader, title):
        """
         Plots the confusion matrix for a model.
         And saves the plot.
        :param eval_loader: torch.utils.data.DataLoader
            Loads the data to evaluate
        :param title: str
            the title of the chart

        Example
        ------------
        >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
        >>> plot_confusion_matrix(dataloader, "title")
        """
        y_pred, y_eval, validation_loss, validation_accuracy = self.eval(eval_loader)
        array = confusion_matrix(y_eval.cpu(), y_pred.cpu(), normalize='true') * 100
        genres = ['country', 'reggae', 'metal', 'pop', 'classical', 'disco', 'hiphop', 'blues', 'jazz', 'rock']
        df_cm = pd.DataFrame(array, index=genres, columns=genres)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu").set_title(title)
        plt.savefig(f"images/{title.replace(' ', '_')}")
        plt.show()

    def classification_report(self, eval_loader):
        """
        Produces the classification_report of the model.
        :param eval_loader: torch.utils.data.DataLoader
            Loads the data to evaluate the chart.
        """
        y_pred, y_eval, validation_loss, validation_accuracy = self.eval(eval_loader)
        genres = ['country', 'reggae', 'metal', 'pop', 'classical', 'disco', 'hiphop', 'blues', 'jazz', 'rock']
        print(classification_report(y_eval.cpu(), y_pred.cpu(), target_names=genres))

    def save(self, name=""):
        with open(f"./samples/trained_models/FbankTrainer_{name}.pkl", 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


class RNNFbankTrainer(FbankTrainer):

    def train(self):
        """
            This model trains a neural network model with producing the
            essential information like training and validation accuracy,
            loss and all the relevant data. So we can use the parent functions
            to reproduce the plots and accuracies per each iteration.
        """
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_losses = []
            epoch_accuracies = []
            for X_train, y_train in self.train_dataloader:
                self.model.zero_grad()
                X_features = self.feature_maker.torch_fbank_features(X_train)
                y_hat = self.model(X_features)
                l = self.loss(y_hat, y_train)
                l.backward()
                self.optimizer.step()
                epoch_losses.append(l.item())
                epoch_accuracies.append(self.accuracy(y_train, torch.argmax(y_hat, dim=1)).item())

            training_loss = sum(epoch_losses) / len(epoch_losses)
            training_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
            self.train_accuracies.append(training_accuracy)
            self.train_losses.append(training_loss)
            y_pred, y_eval, validation_loss, validation_accuracy = self.eval()
            self.validation_losses.append(validation_loss)
            self.validation_accuracies.append(validation_accuracy)
            print(f"============================== EPOCH {epoch + 1} =================================")
            print("Training accuracy %.2f" % (training_accuracy * 100))
            print("Training loss %.4f" % training_loss)
            print("Validation accuracy %.2f" % (validation_accuracy * 100))
            print("Validation loss %.4f" % validation_loss)

    def eval(self, eval_loader=None):
        """
            Evaluates the model by passing the test or validation sets.
        :param eval_loader: torch.utils.data.DataLoader
            Loads the data within the validation or test set to evaluate with
            the trained model.
        :return: torch.Tensor, torch.Tensor, int, int
            the predicted values, the true values, the loss and the accuracy
            of the model would be returned
        """
        self.model.eval()
        if eval_loader is None:
            X_val, y_val = next(iter(self.validation_dataloader))
        else:
            X_val, y_val = next(iter(eval_loader))
        with torch.no_grad():
            X_features = self.feature_maker.torch_fbank_features(X_val)
            y_pred = torch.argmax(self.model(X_features), dim=1)
            l = self.loss(self.model(X_features), y_val)
            accuracy = self.accuracy(y_val, y_pred)
        return y_pred, y_val, l.item(), accuracy.item()


class CNNFbankTrainer(FbankTrainer):

    def train(self):
        """
            This model trains a neural network model with producing the
            essential information like training and validation accuracy,
            loss and all the relevant data. So we can use the parent functions
            to reproduce the plots and accuracies per each iteration.
        """
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_losses = []
            epoch_accuracies = []
            for X_train, y_train in self.train_dataloader:
                self.model.zero_grad()
                X_features = self.feature_maker.torch_fbank_features(X_train)
                y_hat = self.model(X_features)
                l = self.loss(y_hat, y_train)
                l.backward()
                self.optimizer.step()
                epoch_losses.append(l.item())
                epoch_accuracies.append(self.accuracy(y_train, torch.argmax(y_hat, dim=1)).item())

            training_loss = sum(epoch_losses) / len(epoch_losses)
            training_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
            self.train_accuracies.append(training_accuracy)
            self.train_losses.append(training_loss)
            y_pred, y_eval, validation_loss, validation_accuracy = self.eval()
            self.validation_losses.append(validation_loss)
            self.validation_accuracies.append(validation_accuracy)
            print(f"============================== EPOCH {epoch + 1} =================================")
            print("Training accuracy %.2f" % (training_accuracy * 100))
            print("Training loss %.4f" % training_loss)
            print("Validation accuracy %.2f" % (validation_accuracy * 100))
            print("Validation loss %.4f" % validation_loss)

    def eval(self, eval_loader=None):
        """
            Evaluates the model by passing the test or validation sets.
        :param eval_loader: torch.utils.data.DataLoader
            Loads the data within the validation or test set to evaluate with
            the trained model.
        :return: torch.Tensor, torch.Tensor, int, int
            the predicted values, the true values, the loss and the accuracy
            of the model would be returned
        """
        self.model.eval()
        if eval_loader is None:
            X_val, y_val = next(iter(self.validation_dataloader))
        else:
            X_val, y_val = next(iter(eval_loader))
        with torch.no_grad():
            X_features = self.feature_maker.torch_fbank_features(X_val)
            y_pred = torch.argmax(self.model(X_features), dim=1)
            l = self.loss(self.model(X_features), y_val)
            accuracy = self.accuracy(y_val, y_pred)
        return y_pred, y_val, l.item(), accuracy.item()


class SklearnTrainer():
    def __init__(self, models={}, use_pca=True, pca_size=3, use_norm=True) -> None:
        """
          This class can create a number of model instances and train them independently.
          It also can produce the accuracy and the confusion matrix for each model.
        :param models: dict
          of models to train which we can set the type of the model and
          the parameters
        :param use_pca: bool
          defines if we need to use PCA to reduce dimensionality
        :param pca_size: int
          defines the number of features to extract with using PCA
        :param use_norm:
          defines if we need to use min-max normaliztions or not.
          it is recommended to use the normaliztioin.

          Example
          -------
          >>> model = {
          >>> "mlp": {
          >>>    "name": "mlp",
          >>>     "parameters": {
          >>>         "hidden_layer_sizes": (128,128,128,128,128),
          >>>         "solver": "adam",
          >>>         "max_iter": 100,
          >>>         "early_stopping": True,
          >>>          }
          >>>     },
          >>>     "svm_ovo": {
          >>>           "name": "svm",
          >>>           "parameters": {
          >>>             "decision_function_shape":"ovo"
          >>>           }
          >>>     },
          >>>     "svm_ovr": {
          >>>           "name": "svm",
          >>>           "parameters": {
          >>>             "decision_function_shape":"ovr"
          >>>           }
          >>>     },
          >>>     "decision_tree": {
          >>>           "name": "decision_tree",
          >>>           "parameters": {
          >>>           }
          >>>     },
          >>>     "random_forest": {
          >>>           "name": "random_forest",
          >>>           "parameters": {
          >>>           }
          >>>     },
          >>> }

          >>> skl = SklearnTrainer(model, True, 3, True)
        """
        self.models = models
        self.use_pca = use_pca
        self.pca_size = pca_size
        self.use_norm = use_norm
        self.pca_transformer = PCA(pca_size)
        self.min_max_scaler = MinMaxScaler()
        self.models_dict = {}
        self.evaluations = None
        for alias, model in models.items():
            if model["name"].lower() == "svm":
                self.models_dict[alias] = SVC(**model['parameters'])
            elif model["name"].lower() == "decision_tree":
                self.models_dict[alias] = DecisionTreeClassifier(**model['parameters'])
            elif model["name"].lower() == "random_forest":
                self.models_dict[alias] = RandomForestClassifier(**model['parameters'])
            elif model["name"].lower() == "knn":
                self.models_dict[alias] = KNeighborsClassifier(**model['parameters'])
            elif model["name"].lower() == "mlp":
                self.models_dict[alias] = MLPClassifier(**model['parameters'])
            else:
                print(f"No model with name {model['name']} with alias {alias} exists.")

    def train(self, X, y):
        """
        trains all defined model which are defined during the class creation.
        :param X: np.darray
            of features
        :param y: np.darrau
            of classes

        Example
        --------
        >>> X = numpy.array([[1,2,3], [2,3,3]])
        >>> y = numpy.array([1,1])
        >>> train(X, y)
        """
        if self.use_norm:
            self.min_max_scaler.fit_transform(X)
            X = self.min_max_scaler.transform(X)

        if self.use_pca:
            self.pca_transformer.fit(X)
            X = self.pca_transformer.transform(X)

        for alias, model in self.models_dict.items():
            print(f"Training {alias.upper()}")
            print("Model information: ", model)
            model.fit(X, y)

    def eval(self, X_val, y_val):
        """
        Evaluates the model by receiving the X and y and it returns
        a dictionary which contains the accuracy, the predicted y and
        the true y.
        :param X_val: np.darray
            of features
        :param y_val: np.darray
            of classes
        :return: dict
            of accuracy, predictions, and true values

        Example
        --------
        >>> X = numpy.array([[1,2,3], [2,3,3]])
        >>> y = numpy.array([1,1])
        >>> eval(X, y)
        """
        if self.use_norm:
            X_val = self.min_max_scaler.transform(X_val)

        if self.use_pca:
            X_val = self.pca_transformer.transform(X_val)

        self.evaluations = {}
        for name, model in self.models_dict.items():
            print(f"Evaluating {name.upper()}")
            y_pred = model.predict(X_val)
            self.evaluations[name] = {}
            self.evaluations[name]['y_pred'] = y_pred
            self.evaluations[name]['y_val'] = y_val
            self.evaluations[name]['accuracy'] = accuracy_score(y_val, y_pred)

        print("All models are evaluated.")
        return self.evaluations

    def classification_report(self):
        """
        This function generates the classification report using the
        sklearn module.
        """
        if self.evaluations == None:
            raise Exception("First call eval() to obtain preds.")

        for alias, model in self.models_dict.items():
            print(f"CLASSIFICATION REPORT FOR {alias.upper()}:\n")
            genres = ['country', 'reggae', 'metal', 'pop', 'classical', 'disco', 'hiphop', 'blues', 'jazz', 'rock']
            print(classification_report(
                self.evaluations[alias]['y_pred'],
                self.evaluations[alias]['y_val'],
                target_names=genres)
            )

    def plot_confusion_matrix(self):
        """
        Plots the confusion matrix for each trained model and saves
        the results in image folder.
        """
        for alias, model in self.models_dict.items():
            print(f"CONFUSION MATRIX FOR {alias.upper()}:\n")
            array = confusion_matrix(
                self.evaluations[alias]['y_pred'],
                self.evaluations[alias]['y_val'],
                normalize='true') * 100
            genres = ['country', 'reggae', 'metal', 'pop', 'classical', 'disco', 'hiphop', 'blues', 'jazz', 'rock']
            df_cm = pd.DataFrame(array, index=genres, columns=genres)
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True, cmap="YlGnBu").set_title(alias.upper())
            plt.savefig(f"images/{alias.upper()}_Norm_{self.use_norm}_PCA_{self.use_pca}_{self.pca_size}")
            plt.show()


class KmeansTrainer():
    def __init__(self,
                 trained_genres=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 max_iter=5000,
                 n_init=50,
                 random_state=0,
                 use_pca=True,
                 pca_size=200,
                 use_norm=True) -> None:
        """
            This module is a trainer to train kmeans using the selected features
        :param trained_genres: List
            The list of features that we are planning to apply kmeans on it.
        :param max_iter: int
            Maximum number of iterations that we want to run kmeans
        :param n_init: int
            number of initializations
        :param random_state: int
            kmeans random state
        :param use_pca: bool
            Whether we use pca for feature extraction or not.
        :param pca_size: int
            Number of selected components for pca.
        :param use_norm: bool
            Whether to normalize the features or not for kmeans training.
        """
        self.trained_genres = trained_genres
        self.kmeans = KMeans(
            n_clusters=len(trained_genres),
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state)
        self.use_pca = use_pca
        self.pca_size = pca_size
        self.use_norm = use_norm
        self.pca = None
        self.min_max_scaler = None

    def get_data_for_genres(
            self,
            trained_genres,
            X,
            y
    ):
        """
            Here you can choose specific genres to be selected and be used for training
        :param trained_genres: List
            of genres that you are going to apply kmeans on
        :param X: numpy.ndarray
            List of all features to be selected from
        :param y: numpy.ndarray
            List of all targets to be selected from
        :return: numpy.ndarray, numpy.ndarray
            List of all selected features and corresponding targets
        """
        X_selected = []
        y_selected = []
        for i in trained_genres:
            X_selected.append(X[y == i])
            y_selected.append(y[y == i])

        X_out = np.array([i for sub in X_selected for i in sub])
        y_out = np.array([i for sub in y_selected for i in sub])

        return X_out, y_out

    def normalize(self, X):
        """
            Normalizes the parameters using the MinMaxScaler
        :param X: numpy.ndarray
            Features with linear shape
        :return: numpy.ndarray
            Normalized features

        Example
        -------
        >>> X = numpy.array([[1,2,3], [4,5,6]])
        >>> normalize(X)
        """
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.min_max_scaler.fit_transform(X)
        return self.min_max_scaler.transform(X)

    def apply_pca(self, X, n_components):
        """
            Extracts the important features using PCA
        :param X: numpy.ndarray
            Features with linear shape
        :param n_components: int
            Number of expected components
        :return: numpy.ndarray
            Important features values

        Example
        -------
        >>> X = numpy.array([[1,2,3], [4,5,6]])
        >>> apply_pca(X, 10)
        """
        self.pca = PCA(n_components=n_components)
        self.pca.fit_transform(X)
        return self.pca.transform(X)

    def train(self, X, y):
        """
            Trains the kmeans model
        :param X: numpy.ndarray
            Features with linear shape
        :param y: numpy.ndarray
            Targets

        Example
        -------
        >>> X = numpy.array([[1,2,3], [4,5,6]])
        >>> y_true = numpy.array([1,2])
        >>> train(X, y_true)

        """
        X, y = self.get_data_for_genres(self.trained_genres, X, y)
        if self.use_norm:
            X = self.normalize(X)

        if self.use_pca:
            X = self.apply_pca(X, self.pca_size)

        self.kmeans.fit(X)

    def eval(self, X, y):
        """
            Evaluates the kmeans model to see how predictive it is
        :param X: numpy.ndarray
            Features with linear shape
        :param y: numpy.ndarray
            Targets
        :return: numpy.ndarray
            Predictions

        Example
        -------
        >>> X = numpy.array([[1,2,3], [4,5,6]])
        >>> y_true = numpy.array([1,2])
        >>> eval(X, y_true)

        """
        X, y = self.get_data_for_genres(self.trained_genres, X, y)

        if self.use_norm:
            X = self.min_max_scaler.transform(X)

        if self.use_pca:
            X = self.pca.transform(X)

        return self.kmeans.predict(X)

    def adjusted_rand_score(self, X, y):
        """
            Adjusts each cluster in kmeans to a music class.
        :param X: numpy.ndarray
            Features with linear shape
        :param y: numpy.ndarray
            Targets
        :return: float
            adjusted score of kmeans trained model

        Example
        -------
        >>> X = numpy.array([[1,2,3], [4,5,6]])
        >>> y_true = numpy.array([1,2])
        >>> adjusted_rand_score(self, X, y_true)
        """
        X, y = self.get_data_for_genres(self.trained_genres, X, y)

        if self.use_norm:
            X = self.min_max_scaler.transform(X)

        if self.use_pca:
            X = self.pca.transform(X)
        return adjusted_rand_score(y, self.kmeans.predict(X))

    def genres_indicies_to_values(self):
        """
            Converts the number of each genre to their name
        :return: List
            list of names matched with indices
        """
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        return [genres[i] for i in self.trained_genres]

    def plot_adjusted_matrix(self, X, y_true):
        """
            Adjust each class to each cluster to see how well kmeans works
        :param X: numpy.ndarray
            Features in linear shape
        :param y_true: numpy.ndarray
            Targets

        Example
        -------
        >>> X = numpy.array([[1,2,3], [4,5,6]])
        >>> y_true = numpy.array([1,2])
        >>> plot_adjusted_matrix(self, X, y_true)
        """
        X, y = self.get_data_for_genres(self.trained_genres, X, y_true)

        if self.use_norm:
            X = self.min_max_scaler.transform(X)

        if self.use_pca:
            X = self.pca.transform(X)

        matrix = confusion_matrix(y_true, self.kmeans.predict(X))
        empty_rows = []
        for i in range(matrix.shape[0]):
            if (matrix[i].sum() == 0):
                empty_rows.append(i)
        del_count = 0
        for i in range(len(empty_rows)):
            matrix = np.delete(matrix, (empty_rows[i] - del_count), axis=0)
            del_count = del_count + 1

        final_matrix = matrix[:, 0:len(matrix)]
        plt.figure(figsize=(12, 9))
        sn.heatmap(final_matrix, cmap="YlGnBu", annot=True, fmt='d', square=True,
                   xticklabels=[i for i in range(len(final_matrix))],
                   yticklabels=self.genres_indicies_to_values())
        plt.show()

    def accuracy_score(self, X, y_true):
        """
        Calculates the accuracy of the kmeans.
        :param X: numpy.ndarray
            Features in linear form
        :param y_true: numpy.ndarray
            Targets

        Example
        -------
        >>> X = numpy.array([[1,2,3], [4,5,6]])
        >>> y_true = numpy.array([1,2])
        >>> accuracy_score(self, X, y_true)
        """
        print("Adjusted Rand Score: %.3f" % self.adjusted_rand_score(X, y_true))
