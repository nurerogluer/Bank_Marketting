#from deep import Model
from torch.optim import optimizer
#from deep import Model
#from deep_class import DEEP
from torch.optim.optimizer import Optimizer
# from deep import EPOCHS, Model
import pandas as pd
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import tqdm


# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# from torch.autograd import Variable
# import tqdm
# import numpy as np
# import pickle 
data= pd.read_csv('data/bank-additional_test.csv',sep="\t")
X = data.iloc[:, :-1].values
y = data.iloc[:,20].values

ss=StandardScaler()
ss.fit(X)
X=ss.transform(X)

with open('Pickles/classifier_decision_tree.pkl', 'rb') as f:
    loaded_dectree_classifier = pickle.load(f)
y_predDT = loaded_dectree_classifier.predict(X)

print('----------------------')
test_result_DT=loaded_dectree_classifier.score(X,y)
print('Decision Tree Prediction',test_result_DT)
print('----------------------')

with open('Pickles/random_forest_classifier.pkl', 'rb') as f:
     loaded_RF_classifier = pickle.load(f)
y_predRF = loaded_RF_classifier.predict(X)

print('----------------------')
test_result_RF=loaded_RF_classifier.score(X,y)
print('Random Forest Prediction',test_result_RF)
print('----------------------')

with open('Pickles/KN.pkl', 'rb') as f:
    loaded_KN_classifier = pickle.load(f)
y_predKN = loaded_KN_classifier.predict(X)

print('----------------------')
test_result_KN=loaded_KN_classifier.score(X,y)
print('KNeighbors',test_result_KN)
print('----------------------')

with open('Pickles/svm.pkl', 'rb') as f:
     loaded_svm_classifier = pickle.load(f)
y_predsvm = loaded_svm_classifier.predict(X)

print('----------------------')
test_result_svm=loaded_svm_classifier.score(X,y)
print('Support Vektor Machine',test_result_svm)
print('----------------------')

print('----------------------')
