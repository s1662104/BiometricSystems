
#Qui viene realizzato il modello SVM, i vari calcoli come FAR, FRR, HTER,SFAR e il plotting delle curve.

import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, det_curve
from matplotlib import pyplot as plt



class ModelSVM:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    #Modello SVM
    def train_svm(self):
        model = SVC(kernel='rbf', random_state=0, gamma='scale', C=1)

        svm = model.fit(self.X_train, self.y_train)
        #TODO: TOGLIERE QUESTA RIGA DI CODICE
        y_train_score = svm.decision_function(self.X_train)

        y_test_score = svm.predict(self.X_test)

        return svm, y_train_score, y_test_score

#Plot della RocCurve
def plot_roc_curve(y_test, y_test_score):

        FPR, TPR, t = roc_curve(y_test, y_test_score)
        print(FPR)
        print()
        print(TPR)
        print()
        print("TH:",t)
        auc = roc_auc_score(y_test,y_test_score)

        # Plot ROC curve
        plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        sns.set()

def licit_scenario(y_test,y_test_score,index):

        FAR , FRR = calculate_FAR_FRR(y_test,y_test_score,index)

        HTER = ((FAR + FRR) / 2)

        return FRR, FAR, HTER


def spoofing_scenario(y_test,y_test_score,index):
        num_fake = 0
        count = 0
        FAR, FRR = calculate_FAR_FRR(y_test,y_test_score,index)

        if index == 0:

            for i in range(len(y_test)):
                if y_test.iloc[i] == 0:
                    num_fake += 1
                    if y_test.iloc[i] != y_test_score[i]:
                        count+=1

        elif index == 1:
            for i in range(len(y_test)):
                if y_test.iloc[i] == 1:
                    num_fake += 1
                    if y_test.iloc[i] != y_test_score[i]:
                        count+=1
        else:
            exit("WRONG INDEX")
        SFAR = count / num_fake

        return FRR,SFAR

#Calcolo di FAR e FRR
def calculate_FAR_FRR(y_test,y_test_score, index):

    FA = 0
    FR = 0
    num_fake = 0
    num_real = 0
    if index == 0:
        for i in range(len(y_test)):

            if y_test.iloc[i] == 0:
                num_fake+=1
            if y_test.iloc[i] == 1:
                num_real+=1

        for i in range(len(y_test)):
            if y_test.iloc[i] == 1:
                if y_test.iloc[i] != y_test_score[i]:
                    FR+=1

            if y_test.iloc[i] == 0:
                if y_test.iloc[i] != y_test_score[i]:
                    FA += 1
    elif index == 1:
        for i in range(len(y_test)):

            if y_test.iloc[i] == 1:
                num_fake += 1
            if y_test.iloc[i] == 0 :
                num_real += 1

        for i in range(len(y_test)):
            if y_test.iloc[i] == 0:
                if y_test.iloc[i] != y_test_score[i]:
                    FR += 1

            if y_test.iloc[i] == 1:
                if y_test.iloc[i] != y_test_score[i]:
                    FA += 1
    else:
        exit("WRONG INDEX")

    FAR = FA / num_fake
    FRR = FR / num_real
    return FAR, FRR

#Plot della det_curve
def plot_det_curve(y_test,y_test_score):
    FAR, FRR, t = det_curve(y_test, y_test_score)
    print("FAR",FAR)
    print("FRR",FRR)
    print("TH:",t)

    plt.plot(FAR, FRR, label='DET curve')
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('Detection Error Tradeoff ')
    plt.legend(loc="lower right")
    plt.show()
    sns.set()