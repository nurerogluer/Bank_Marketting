from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from subprocess import check_call
import pickle 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


#from preprocces import X_train,X_test,y_train,y_test
class Dectree:
    def __init__(self,X_train, X_test, y_train, y_test):
         
        self.dt_train = tree.DecisionTreeClassifier(max_depth=7, random_state=0, class_weight='balanced')
        self.dt_train=self.dt_train.fit(X_train,y_train)
        self.dt_test_result=self.dt_train.score(X_test,y_test)     
        #prediciton decisiontree
        y_pred =self.dt_train.predict(X_test)
        
                
        dot_data = tree.export_graphviz(self.dt_train, out_file='Figuren/Tree.dot')
        check_call(['dot','-Tpng','Figuren/tree.dot','-o','Figuren/Decision_tree.png'])
      #  tree.plot_tree(self.dt_train)
    
        #print('Score Decision Tree', self.dt_test_result)
 

        with open('Pickles/classifier_decision_tree.pkl', 'wb') as f:
            pickle.dump(self.dt_train, f)

        #Create the matrix that shows how often predicitons were done correctly and how often theey failed.
        conf_mat = confusion_matrix(y_test, y_pred)
        #The diagonal ones are the correctly predicted instances. The sum of this number devided by the number of all instances gives us the accuracy in percent.
        accuracy = (conf_mat[0,0] + conf_mat[1,1]) /(conf_mat[0,0]+conf_mat[0,1]+ conf_mat[1,0]+conf_mat[1,1])
        #Create the matrix that shows how often predicitons were done correctly and how often theey failed.

        from sklearn.metrics import ConfusionMatrixDisplay
        fig1=plt.figure()
        self.cm_display = ConfusionMatrixDisplay(conf_mat).plot(cmap='plasma')
        plt.savefig('Figuren/conf_Dectree.png', bbox_inches='tight')

        
        print(' ')
        print('-----------------------------')
        print('       Decision Tree         ')      
        print('-----------------------------')
        print(' ')
        print('Accuracy Decision Tree: ' + str(round(accuracy,4)))
        print(' ')
        print('Confusion matrix Decision Tree:')
        print(conf_mat)
        print(' ')  
        print('classification report Decision Tree:')
        print(classification_report(y_test, y_pred)) 
        print('fertig trainiert Decision Tree')
        print('-----------------------------')

        #########
        ## ROC ##
        #########
        self.probs=self.dt_train.predict_proba(X_test)
        self.probs = self.probs[:, 1]
        self.auc = roc_auc_score(y_test, self.probs)
        print('AUC: %.2f' % self.auc)
        self.fpr, self.tpr, self.thresholds = roc_curve(y_test, self.probs)
       # def plot_roc_curve(fpr, tpr):
        fig2=plt.figure()
        #plt.plot(fpr, tpr, color='orange', label='AUC = {:.3f}'.format(auc(fpr, tpr)),label='ROC')
        plt.plot(self.fpr, self.tpr, color='orange')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        #plt.legend();
        #plt.show()
                #plot_roc_curve(self.fpr, self.tpr)
        fig2.savefig('Figuren/roc_Dectree.png', bbox_inches='tight')
     
    def get_data_tree1(self):
        return self.fpr
    def get_data_tree2(self):
        return self.tpr
    def get_data_tree3(self):
        return self.dt_test_result
 








 



    

