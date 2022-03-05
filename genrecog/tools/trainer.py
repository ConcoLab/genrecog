from genrecog.preprocess.feature import Feature
import torch


class FbankTrainer:
    def __init__(self, model, optimizer, loss, dataloader, num_epochs, X_val, y_val):
        self.model = model
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.loss = loss
        self.train_losses = []
        self.validation_losses = []
        self.feature_maker = Feature()
        self.X_val = X_val
        self.y_val = y_val

    def train(self):
        print("Training is started.")
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_losses = []
            for X_train, y_train in self.dataloader:
                self.model.zero_grad()
                X_features = self.feature_maker.torch_fbank_features(X_train).transpose(1, -1)
                y_hat = self.model(X_features)
                l = self.loss(y_hat, y_train)
                l.backward()
                self.optimizer.step()
                epoch_losses.append(l.item())
                print("Epoch %2d final minibatch had loss %.4f" % (epoch, l.item()))
            self.train_losses.append(sum(epoch_losses) / len(epoch_losses))
            print(self.train_losses)
            print(epoch)
            y_pred, validation_loss = self.eval()
            print(validation_loss)
            self.validation_losses.append(validation_loss)

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            X_features = self.feature_maker.torch_fbank_features(self.X_val).transpose(1, -1)
            y_pred = torch.softmax(self.model(X_features), dim=1)
            l = self.loss(y_pred, self.y_val)
        return y_pred, l.item()
