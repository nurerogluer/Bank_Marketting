import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
import pickle 
from sklearn.preprocessing import StandardScaler
data= pd.read_csv('data/bank.csv',sep="\t")
X = data.iloc[:, :-1].values
y = data.iloc[:,20].values
#print (y)
ss=StandardScaler()
ss.fit(X)
X=ss.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print(' X_train, y_train, X_test, y_test ')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



#****DECISION TREE*****
from train_decisiontree import Dectree
DT=Dectree(X_train, X_test, y_train, y_test) 
DT_fpr=DT.get_data_tree1()
DT_tpr=DT.get_data_tree2()
DT_scr=DT.get_data_tree3()

#****RANDOM FOREST****
from train_randomforest import Randtree
RF=Randtree(X_train, X_test, y_train, y_test)
RF_fpr=RF.get_data_rand1()
RF_tpr=RF.get_data_rand2()
RF_scr=RF.get_data_rand3()

#****KNN****
from train_knn import Knn
kn=Knn(X_train, X_test, y_train, y_test)
kn_fpr=kn.get_data_knn1()
kn_tpr=kn.get_data_knn2()
kn_scr=kn.get_data_knn3()

#****SVM****
from train_svm import Svm
SVM=Svm(X_train, X_test, y_train, y_test)
SVM_fpr=SVM.get_data_svm1()
SVM_tpr=SVM.get_data_svm2()
SVM_scr=SVM.get_data_svm3()


#****DEEP LEARNING****
from deep_class import DEEP
DPL=DEEP(X_train, X_test, y_train, y_test)
DPL_fpr=DPL.get_data_deep1()
DPL_tpr=DPL.get_data_deep2()


print('Score Decision Tree=', DT_scr)
print('Score Random Forest=', RF_scr)
print('Score KNN=', kn_scr)
print('Score SVM=', SVM_scr)

plt.plot(SVM_fpr, SVM_tpr, color='black', label='SVM_TREE')
plt.plot(DT_fpr, DT_tpr, color='orange', label='DEC_TREE')
plt.plot(RF_fpr, RF_tpr, color='lightblue', label='RAND_FOR')
plt.plot(kn_fpr, kn_tpr, color='green', label='KNN')
plt.plot(DPL_fpr, DPL_tpr,color='red', label='DEEP')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
# #plt.show()

plt.savefig('Figuren/ROCS.png', bbox_inches='tight') 