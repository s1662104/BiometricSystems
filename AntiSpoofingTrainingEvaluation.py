import csv
from typing import Any, Union

import matplotlib

import Antispoofing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt
import numpy as np
import os


class ModelSVM:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_svm(self):
        model = SVC(kernel='rbf', random_state=0, gamma='scale', C=1)
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
        print(FPR)
        print()
        print(TPR)
        print()
        print(t)
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
        FRR = 1 - TPR
        SFAR = FPR


        return FRR,SFAR


def plot_FAR_FRR_ERR(FAR,FRR,threshold):
    fig, ax = plt.subplots()

    ax.plot(threshold, FAR, 'r--', label='FAR')
    ax.plot(threshold, FRR, 'g--', label='FRR')
    plt.xlabel('Threshold')
    #plt.plot(15, EER, 'ro', label='EER')

    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')

    plt.show()

def calculate_FAR_FRR_ERR(y_test,y_test_score):
    FAR =[]
    FRR =[]
    FA = 0
    FR = 0
    num_fake = 0
    num_real = 0
    for i in range(len(y_test)):
        if y_test[i] == 0:
            num_fake+=1
        if y_test[i] == 1:
            num_real+=1

    for i in range(len(y_test)):
        if y_test[i] == 1: ##Utente Real
            if y_test[i] != y_test_score[i]:
                FR+=1
                FRR.append(FR / num_real)
        if y_test[i] == 0:
            if y_test[i] != y_test_score[i]:
                FA += 1
                FAR.append(FA / num_fake)

    FAR = sorted(FAR)
    FRR = sorted(FRR, reverse = True)
    return FAR, FRR



def plot_det_curve(y_true,y_score):
    FPR, FNR, t = roc_curve(y_true, y_score)
    print("FPR",FPR)
    print("FNR",FNR)
    print(t)
    axis_min = min(FPR[0], FNR[-1])
    fig, ax = plt.subplots()
    plt.plot(FPR, FNR)
    plt.yscale('log')
    plt.xscale('log')
    ticks_to_use = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    plt.axis([0.001, 50, 0.001, 50])
    plt.show()
