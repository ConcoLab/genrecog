"""Library to train models and generate specific reports
Authors
 * Amirali Ashraf 2020
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sn


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

    def plot_accuracies(self, title):
        """
        plots the accuracies per iteration
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

    def plot_confusion_matrix(self, eval_loader, title):
        """
         Plots the confusion matrix for a model.
         And saves the plot.
        :param eval_loader: torch.utils.data.DataLoader
            Loads the data to evaluate
        :param title: str
            the title of the chart
        """
        y_pred, y_eval, validation_loss, validation_accuracy = self.eval(eval_loader)
        array = confusion_matrix(y_eval.cpu(), y_pred.cpu(), normalize='true') * 100
        genres = ['country', 'reggae', 'metal', 'pop', 'classical', 'disco', 'hiphop', 'blues', 'jazz', 'rock']
        df_cm = pd.DataFrame(array, index=genres, columns=genres)
        plt.figure(figsize=(10, 7)).set_title(title)
        plt.savefig(f"images/{title.replace(' ', '_')}")
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu")


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
        model = {
        "mlp": {
            "name": "mlp",
            "parameters": {
                "hidden_layer_sizes": (128,128,128,128,128),
                "solver": "adam",
                "max_iter": 100,
                "early_stopping": True,
                 }
            },
            "svm_ovo": {
                  "name": "svm",
                  "parameters": {
                    "decision_function_shape":"ovo"
                  }
            },
            "svm_ovr": {
                  "name": "svm",
                  "parameters": {
                    "decision_function_shape":"ovr"
                  }
            },
            "decision_tree": {
                  "name": "decision_tree",
                  "parameters": {
                  }
            },
            "random_forest": {
                  "name": "random_forest",
                  "parameters": {
                  }
            },
        }
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


