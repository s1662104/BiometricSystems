from typing import Any, Union

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from mlxtend.plotting import plot_decision_regions
import numpy as np


class ModelSVM:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_svm(self):
        model = SVC(kernel='rbf', random_state=0, gamma=1, C=1)
        # new code
        # pca = PCA(n_components = 2)
        # X_train2 = pca.fit_transform(X_train)
        # end code
        svm = model.fit(self.X_train, self.y_train)  # Transform X_train to X_train2
        y_train_score = svm.decision_function(self.X_train)

        # FPR, TPR, t = roc_curve(y_train, y_train_score)
        # roc_auc = auc(FPR, TPR)

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # ax1.plot(FPR, TPR, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc, color='b')
        # ax1.set_title('Training Data')

        y_test_score = svm.decision_function(self.X_test)
        return svm, y_train_score, y_test_score

###FUNZIONE TEMPORANEA

def plot_roc_curve(y_test, y_test_score):
        FPR, TPR, t = roc_curve(y_test, y_test_score)
        roc_auc = auc(FPR, TPR)

        # fig, ax1 = plt.subplots()
        # ax1.plot(FPR, TPR, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc, color='b')
        # ax1.set_title('Test Data')

        plt.figure()
        lw = 2
        plt.plot(FPR[1], TPR[1], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        #Non plotta a roc curve...



        # for ax in fig.axes:
        #     ax.plot([0, 1], [0, 1], 'k--')
        #     ax.set_xlim([-0.05, 1.0])
        #     ax.set_ylim([0.0, 1.05])
        #     ax.set_xlabel('False Positive Rate')
        #     ax.set_ylabel('True Positive Rate')
        #     ax.legend(loc="lower right")
        #
        # plt.show()

def licit_scenario(y_test,y_test_score):

        FPR, TPR, t = roc_curve(y_test, y_test_score)
        ###EER Parte
        # fpr, tpr, threshold = roc_curve(y_train, y_pred, pos_label=1)
        fnr = 1 - TPR
        # eer_threshold = t[np.nanargmin(np.absolute((fnr - FPR)))]
        #
        # print("EER_THRESHOLD: ", eer_threshold)
        #
        # EER = FPR[np.nanargmin(np.absolute((fnr - FPR)))]
        #
        # print("EER:", EER)
        #
        # EER = fnr[np.nanargmin(np.absolute((fnr - FPR2)))]
        #
        # print("EER: ", EER)

        ### FAR + FRR + HTER

        # print("#### LICIT SCENARIO ####")
        # FRR = FN / (TP + FN) = 1 - TPR
        # FAR = FP / (FP + TN) = FPR
        FRR = 1 - TPR
        FAR = FPR
        # print("FRR: ",FRR)
        # print("FAR: ", FAR)

        HTER = ((FAR + FRR) / 2)

        return FRR, FAR, HTER


def spoofing_scenario(y_test,y_test_score):

        FPR, TPR, t = roc_curve(y_test, y_test_score)
        fnr = 1 - TPR
        FRR = 1- TPR
        SFAR = FPR


        return FRR,SFAR