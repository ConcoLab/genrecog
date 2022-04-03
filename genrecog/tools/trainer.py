from genrecog.preprocess.feature import Feature
import torch
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle


class FbankTrainer():
    def __init__(self, model, optimizer, loss, train_dataloader, validation_dataloader, num_epochs):
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
        return (torch.sum(y_true == y_pred) / y_pred.shape[0])

    def plot_loss(self):
        plt.plot(self.train_losses)
        plt.plot(self.validation_losses)
        plt.legend(['Training loss', 'Validation loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')

    def plot_accuracies(self):
        plt.plot(self.train_accuracies)
        plt.plot(self.validation_accuracies)
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.xlabel('epoch')
        plt.ylabel('accuracy %')

    def plot_confusion_matrix(self, eval_loader):
        y_pred, y_eval, validation_loss, validation_accuracy = self.eval(eval_loader)
        array = confusion_matrix(y_eval.cpu(), y_pred.cpu(), normalize='true') * 100
        genres = ['country', 'reggae', 'metal', 'pop', 'classical', 'disco', 'hiphop', 'blues', 'jazz', 'rock']
        df_cm = pd.DataFrame(array, index=genres, columns=genres)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu")

    def classification_report(self, eval_loader):
        y_pred, y_eval, validation_loss, validation_accuracy = self.eval(eval_loader)
        genres = ['country', 'reggae', 'metal', 'pop', 'classical', 'disco', 'hiphop', 'blues', 'jazz', 'rock']
        print(classification_report(y_eval.cpu(), y_pred.cpu(), target_names=genres))

    def save(self, name=""):
        with open(f"./samples/trained_models/FbankTrainer_{name}.pkl", 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


class RNNFbankTrainer(FbankTrainer):

    def train(self):
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


