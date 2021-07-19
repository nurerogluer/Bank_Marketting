from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from subprocess import check_call
import pickle 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class Svm:
    def __init__(self,X_train, X_test, y_train, y_test):
        
        self.sc_X = StandardScaler()
        self.X_train = self.sc_X.fit_transform(X_train)
        self.X_test = self.sc_X.transform(X_test)
       
        self.classifier_svm = SVC(kernel='poly', random_state = 0, class_weight='balanced',probability=True)
        self.dd=self.classifier_svm.fit(X_train, y_train)
                
        self.svm_test_result=self.classifier_svm.score(X_test,y_test)
        self.y_pred = self.classifier_svm.predict(X_test)

        
        #Create the matrix that shows how often predicitons were done correctly and how often theey failed.
        conf_mat_svm = confusion_matrix(y_test,self.y_pred)
        #The diagonal ones are the correctly predicted instances. The sum of this number devided by the number of all instances gives us the accuracy in percent.
        accuracy_svm= (conf_mat_svm[0,0] + conf_mat_svm[1,1]) /(conf_mat_svm[0,0]+conf_mat_svm[0,1]+ conf_mat_svm[1,0]+conf_mat_svm[1,1])
        
        from sklearn.metrics import ConfusionMatrixDisplay
        plt.figure()
        self.cm_display = ConfusionMatrixDisplay(conf_mat_svm).plot(cmap='plasma')
        plt.savefig('Figuren/conf_SVM.png', bbox_inches='tight')

        print(' ')
        print('-----------------------------')
        print('    Support Vektor Machine    ')      
        print('-----------------------------')
        print(' ')
        
        print('Accuracy SVM: ' + str(round(accuracy_svm,4)))
        print('Confusion matrix SVM:')
        print(conf_mat_svm)  
        print('classification report SVM:')
        print(classification_report(y_test, self.y_pred)) 

        print('fertig trainiert SVM')
        print('-----------------------------')
    
        with open('Pickles/svm.pkl', 'wb') as f:
             pickle.dump(self.classifier_svm, f)

        #########
        ## ROC ##
        #########
        self.probs=self.classifier_svm.predict_proba(X_test)
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
        fig2.savefig('Figuren/roc_SVM.png', bbox_inches='tight')

    def get_data_svm1(self):
        return self.fpr
    def get_data_svm2(self):
        return self.tpr
    def get_data_svm3(self):
        return self.svm_test_result


    def get_data_svm(self):
        return self.classifier_svm 