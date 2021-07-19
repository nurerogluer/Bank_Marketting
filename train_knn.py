
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from subprocess import check_call
import pickle 
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn import metrics

class Knn:
    def __init__(self,X_train, X_test, y_train, y_test):
       
        self.classifier = KNeighborsClassifier(n_neighbors=25,  metric='minkowski', p = 2)
        self.cc=self.classifier.fit(X_train, y_train)
        self.KN_test_result=self.classifier.score(X_test,y_test)
        print("KN(test): ",round(self.KN_test_result,2))

        y_pred_KN= self.classifier.predict(X_test)


        with open('Pickles/KN.pkl', 'wb') as f:
             pickle.dump(self.classifier, f)
        #Create the matrix that shows how often predicitons were done correctly and how often theey failed.
        conf_mat_KN = confusion_matrix(y_test, y_pred_KN)
        #The diagonal ones are the correctly predicted instances. The sum of this number devided by the number of all instances gives us the accuracy in percent.
        accuracy_KN= (conf_mat_KN[0,0] + conf_mat_KN[1,1]) /(conf_mat_KN[0,0]+conf_mat_KN[0,1]+ conf_mat_KN[1,0]+conf_mat_KN[1,1])
        from sklearn.metrics import ConfusionMatrixDisplay
        plt.figure()
        self.cm_display = ConfusionMatrixDisplay(conf_mat_KN).plot(cmap='plasma')
        plt.savefig('Figuren/conf_KN.png', bbox_inches='tight')  

        print(' ')
        print('-----------------------------')
        print('       KNeighbors         ')      
        print('-----------------------------')
        print(' ')
        
        print('Accuracy KNN: ' + str(round(accuracy_KN,4)))
        print('Confusion matrix KNeighbors:')
        print(conf_mat_KN)  
        print('classification report KNeighbors:')
        print(classification_report(y_test, y_pred_KN)) 

        print('fertig trainiert KNeighbors')
        print('-----------------------------')
        f1_score(y_test,y_pred_KN)
        error = []

        # Calculating error for K values between 1 and 40
        for i in range(1, 40):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error.append(np.mean(pred_i != y_test))
        print('-----------------------------')
        fig2=plt.figure()
        #plt.figure(figsize=(12, 6))
        plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
        plt.title('Error Rate K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        #plt.show()
        fig2.savefig('Figuren/knn.png', bbox_inches='tight')
        #########
        ## ROC ##
    #     #########
        self.probs=self.classifier.predict_proba(X_test)
        self.ypred_probs = self.probs[:, 1]
        self.auc = roc_auc_score(y_test, self.ypred_probs)
        print('AUC: %.2f' % self.auc)
        #self.fpr, self.tpr, self.thresholds = metrics.roc_curve(y_test, self.ypred_probs)
        self.fpr, self.tpr, self.thresholds = roc_curve(y_test, self.ypred_probs)
 
        #self.roc_auc=metrics.auc(self.fpr, self.tpr)
        #print(self.roc_auc)
        fig3=plt.figure()
        plt.plot(self.fpr, self.tpr,color='orange' )
        
        #plt.plot(self.fpr, self.tpr,'o',label = 'Normalized data: AUC =  ' + str(round(self.roc_auc,4)))
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc = 'lower right');
        #plt.show()
                #plot_roc_curve(self.fpr, self.tpr)
        fig3.savefig('Figuren/roc_KNN.png', bbox_inches='tight')


    def get_data_knn1(self):
        return self.fpr
    def get_data_knn2(self):
        return self.tpr
    def get_data_knn3(self):
        return self.KN_test_result
   

        