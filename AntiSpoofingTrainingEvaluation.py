# Qui viene realizzato il modello SVM, i vari calcoli come FAR, FRR, HTER,SFAR e il plotting delle curve.

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

    # Modello SVM
    def train_svm(self):

        #viene creato il modello
        model = SVC(kernel='rbf', random_state=0, gamma='scale', C=1)

        #viene addestrato
        svm = model.fit(self.X_train, self.y_train)

        # ci da la sua predizione in fase test
        y_test_score = svm.predict(self.X_test)

        #viene ritornato il classificatore addestrato e la predizione
        return svm, y_test_score


# Plot della Roc-Curve
def plot_roc_curve(y_test, y_test_score):
    FPR, TPR, t = roc_curve(y_test, y_test_score)
    print(FPR)
    print()
    print(TPR)
    print()
    print("TH:", t)
    auc = roc_auc_score(y_test, y_test_score)

    # Plot ROC curve
    plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FAR')
    plt.ylabel('GAR')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    sns.set()





# qui vengono calcolati i SFAR e i FRR, viene utilizzato index perché microtexture (index == 1) vede real con
# valore 0 e i fake con valore 1, mentre eyeblink (index == 0) vede real con valore 1 e fake con valore 0.
def spoofing_scenario(y_test, y_test_score, index):
    num_fake = 0
    count = 0

    # andiamo a prenderci il FRR
    _, FRR = calculate_FAR_FRR(y_test, y_test_score, index)

    # per eyeblink
    if index == 0:
        # andiamo a calcolare i FA (false acceptance)
        for i in range(len(y_test)):
            if y_test.iloc[i] == 0:
                num_fake += 1
                if y_test.iloc[i] != y_test_score[i]:
                    count += 1
    # per microtexture
    elif index == 1:
        # andiamo a caclolare i FA (false acceptance)
        for i in range(len(y_test)):
            if y_test.iloc[i] == 1:
                num_fake += 1
                if y_test.iloc[i] != y_test_score[i]:
                    count += 1
    else:
        exit("WRONG INDEX")

    # andiamo a calcolare lo spoof false acceptance rate (SFAR)
    SFAR = count / num_fake

    return FRR, SFAR


# Calcolo di FAR e FRR
# qui vengono calcolato FAR e FRR, viene utilizzato index perché microtexture (index == 1) vede real con
# valore 0 e i fake con valore 1, mentre eyeblink (index == 0) vede real con valore 1 e fake con valore 0.
def calculate_FAR_FRR(y_test, y_test_score, index):
    FA = 0
    FR = 0
    num_fake = 0
    num_real = 0
    # per eyeblink
    if index == 0:

        #contiamo i real e i fake
        for i in range(len(y_test)):

            if y_test.iloc[i] == 0:
                num_fake += 1
            if y_test.iloc[i] == 1:
                num_real += 1

        # contiamo i FR (False rejection)
        for i in range(len(y_test)):
            if y_test.iloc[i] == 1:
                if y_test.iloc[i] != y_test_score[i]:
                    FR += 1
        # contiamo i FA ( False acceptance)
            if y_test.iloc[i] == 0:
                if y_test.iloc[i] != y_test_score[i]:
                    FA += 1
    # per microtexture
    elif index == 1:

        # contiamo i real e i fake
        for i in range(len(y_test)):

            if y_test.iloc[i] == 1:
                num_fake += 1
            if y_test.iloc[i] == 0:
                num_real += 1
        # contiamo i FR (False rejection)
        for i in range(len(y_test)):
            if y_test.iloc[i] == 0:
                if y_test.iloc[i] != y_test_score[i]:
                    FR += 1
        # contiamo i FA ( False acceptance)
            if y_test.iloc[i] == 1:
                if y_test.iloc[i] != y_test_score[i]:
                    FA += 1
    else:
        exit("WRONG INDEX")

    # calcolo di FAR e FRR
    FAR = FA / num_fake
    FRR = FR / num_real
    return FAR, FRR



