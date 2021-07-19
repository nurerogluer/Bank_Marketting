from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from subprocess import check_call
import pickle 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class Randtree:
    def __init__(self,X_train, X_test, y_train, y_test):

        self.clf_rf = RandomForestClassifier(n_estimators=30, max_depth=6,
                    min_samples_split=2, random_state=0, class_weight='balanced')
    
        self.clf_rf.fit(X_train, y_train)
        self.df_test_result=self.clf_rf.score(X_test,y_test)
        self.y_pred = self.clf_rf.predict(X_test)
       
        #n_estimators:wieviel tree default "100"
        #Create the matrix that shows how often predicitons were done correctly and how often theey failed.
        conf_mat_rand = confusion_matrix(y_test,self.y_pred)
        #The diagonal ones are the correctly predicted instances. The sum of this number devided by the number of all instances gives us the accuracy in percent.
        accuracy_rand= (conf_mat_rand[0,0] + conf_mat_rand[1,1]) /(conf_mat_rand[0,0]+conf_mat_rand[0,1]+ conf_mat_rand[1,0]+conf_mat_rand[1,1])
        from sklearn.metrics import ConfusionMatrixDisplay
        plt.figure()
        self.cm_display = ConfusionMatrixDisplay(conf_mat_rand).plot(cmap='plasma')
        plt.savefig('Figuren/conf_RandFor.png', bbox_inches='tight')
        print(' ')
        print('-----------------------------')
        print('       Random Forest         ')      
        print('-----------------------------')
        print(' ')
        
        print('Accuracy Random Forest: ' + str(round(accuracy_rand,4)))
        print('Confusion matrix Random Forest:')
        print(conf_mat_rand)  
        print('classification report Random Forest:')
        print(classification_report(y_test, self.y_pred)) 

        print('fertig trainiert Random Forest')
        print('-----------------------------')
    
        with open('Pickles/random_forest_classifier.pkl', 'wb') as f:
             pickle.dump(self.clf_rf, f)
        #########
        ## ROC ##
        #########
        self.probs=self.clf_rf.predict_proba(X_test)
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
        plt.legend();
        #plt.show()
                #plot_roc_curve(self.fpr, self.tpr)
        fig2.savefig('Figuren/roc_RanFor.png', bbox_inches='tight')

    def get_data_rand1(self):
        return self.fpr
    def get_data_rand2(self):
        return self.tpr

    def get_data_rand3(self):
        return self.df_test_result
    # def get_data_rand2(self):
    #     return self.clf_rf
