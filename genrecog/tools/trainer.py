from genrecog.preprocess.feature import Feature
import torch


class FbankTrainer:
    def __init__(self, model, optimizer, loss, train_dataloader, eval_loader, num_epochs):
        self.model = model
        self.train_dataloader = train_dataloader
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.loss = loss
        self.train_losses = []
        self.validation_losses = []
        self.feature_maker = Feature()
        self.eval_loader = eval_loader

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_losses = []
            for X_train, y_train in self.train_dataloader:
                self.model.zero_grad()
                X_features = self.feature_maker.torch_fbank_features(X_train).transpose(1, -1)
                X_features = torch.nn.functional.normalize(X_features)
                y_hat = self.model(X_features)
                l = self.loss(y_hat, y_train)
                l.backward()
                self.optimizer.step()
                epoch_losses.append(l.item())
                print("Epoch %2d final minibatch had loss %.4f" % (epoch, l.item()))
            self.train_losses.append(sum(epoch_losses) / len(epoch_losses))
            y_pred, y_eval, validation_loss = self.eval()
            print("Epoch %2d final minibatch had test loss %.4f" % (epoch, validation_loss))
            self.validation_losses.append(validation_loss)

    def eval(self, eval_loader=None):
        self.model.eval()
        if eval_loader is None:
            X_val, y_val = next(iter(self.eval_loader))
            with torch.no_grad():
                X_features = self.feature_maker.torch_fbank_features(X_val).transpose(1, -1)
                X_features = torch.nn.functional.normalize(X_features)
                y_pred = torch.softmax(self.model(X_features), dim=1)
                l = self.loss(y_pred, y_val)
            return y_pred, y_val, l.item()
        else:
            X_val, y_val = next(iter(eval_loader))
            with torch.no_grad():
                X_features = self.feature_maker.torch_fbank_features(X_val).transpose(1, -1)
                X_features = torch.nn.functional.normalize(X_features)
                y_pred = torch.softmax(self.model(X_features), dim=1)
                l = self.loss(y_pred, y_val)
            return y_pred, y_val, l.item()
