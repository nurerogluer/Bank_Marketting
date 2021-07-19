
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import tqdm
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler
import pandas as pd
class DEEP:
    def __init__(self,X_train, X_test, y_train, y_test):
        # Get cpu or gpu device for training.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))
        class Model(torch.nn.Module):
            def __init__(self, input_dim):
                super(Model, self).__init__()
                self.fc1 = nn.Linear(input_dim,30)
                self.fc2 = nn.Linear(30,2)
            def forward(self,x):
                x = torch.relu(self.fc1(x))
                x = torch.sigmoid(self.fc2(x))
                return x   
        self.model = Model(X_train.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn= nn.CrossEntropyLoss()
        EPOCHS  = 1000
        X_train = Variable(torch.from_numpy(X_train)).float()
        y_train = Variable(torch.from_numpy(y_train)).long()
        X_test  = Variable(torch.from_numpy(X_test)).float()
        y_test  = Variable(torch.from_numpy(y_test)).long()

        loss_list     = np.zeros((EPOCHS,))
        accuracy_list = np.zeros((EPOCHS,))

        for epoch in tqdm.trange(EPOCHS):
            y_pred = self.model(X_train)
            loss = loss_fn(y_pred, y_train)
            loss_list[epoch] = loss.item()
    
            # Zero gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            with torch.no_grad():
                y_pred = self.model(X_test)
                correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
                accuracy_list[epoch] = correct.mean()
        print('    ')    
        print('Accuracy Deep test',accuracy_list[epoch])
        self.scr=accuracy_list[epoch]
        # with open('Pickles/deep.pkl', 'wb') as f:
        #      pickle.dump(self.model, f)
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

        ax1.plot(accuracy_list)
        ax1.set_ylabel("validation accuracy")
        ax2.plot(loss_list)
        ax2.set_ylabel("validation loss")
        ax2.set_xlabel("epochs");

        plt.savefig('Figuren/deep.png', bbox_inches='tight')

        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import OneHotEncoder

        plt.figure(figsize=(10, 10))
        plt.plot([0, 1], [0, 1], 'k--')

        # One hot encoding
        enc = OneHotEncoder()
        Y_onehot = enc.fit_transform(y_test[:, np.newaxis]).toarray()

        with torch.no_grad():
            y_pred = self.model(X_test).numpy()
            self.fpr, self.tpr, threshold = roc_curve(Y_onehot.ravel(), y_pred.ravel())

        plt.plot(self.fpr, self.tpr, color='orange', label='AUC = {:.3f}'.format(auc(self.fpr, self.tpr)))
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--') 
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend();
        plt.savefig('Figuren/roc_deep.png', bbox_inches='tight')

        data2= pd.read_csv('data/bank-additional_test.csv',sep="\t")
        XX = data2.iloc[:, :-1].values
        yy = data2.iloc[:,20].values
        sss=StandardScaler()
        sss.fit(XX)
        XX=sss.transform(XX)
        XX  = Variable(torch.from_numpy(XX)).float()
        yy  = Variable(torch.from_numpy(yy)).long()

        # Disable grad
        with torch.no_grad():
            ypredxx=self.model(XX)
 
            correctxx = (torch.argmax(ypredxx, dim=1) == yy).type(torch.FloatTensor)
            self.accuracyxx=correctxx.mean()
            #lossxx = loss_fn(ypredxx, yy)
  
            print('Accuracy predicton bank_additional_test',self.accuracyxx)    
    def get_data_deep1(self):
        return self.fpr
    def get_data_deep2(self):
        return self.tpr

