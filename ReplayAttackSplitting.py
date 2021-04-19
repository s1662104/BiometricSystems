import csv
import sys
import os
import numpy as np
import shutil
import cv2
import pandas as pd
import AntiSpoofingTrainingEvaluation


import LBP

#import dask
#import dask.array as da
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from mlxtend.plotting import plot_decision_regions
#from dask_ml.model_selection import train_test_split

csv.field_size_limit(sys.maxsize)


def get_normalized(image):
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
    return norm_image








def column_len_csv(filecsv):
    with open(filecsv, 'r') as f:
        reader = csv.reader(f,delimiter=';')
        for row in reader:
             return len(row)

#Qui viene effettuato lo splitting del dataset nel csv per il replay-attack
class ReplayAttackSplitting():
    def __init__(self,nomeFileCsv):
        self.nomeFileCsv = nomeFileCsv

    def splitting_train_test(self,filecsv):
        num_columns = column_len_csv(filecsv)
        print(num_columns)

        data = pd.read_csv(filecsv, sep=';', header= None)


        X,y = data.iloc[:, :-1], data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1, shuffle=True)

        return X_train, X_test, y_train, y_test

    def writeCsv(self, nameFileCsv, histogram, val):
        print(histogram)
        list = []
        for val_array in histogram:
            list.append(val_array)
        list.append(val)

        with open(nameFileCsv, 'a+') as cvsfile:
            writer = csv.writer(cvsfile, delimiter=';')
            writer.writerow(list)
            cvsfile.close()

    def convert_image_to_hist(self,image):
        print(image)
        image = cv2.imread(image)

        norm_image = get_normalized(image)

        myLBP = LBP.Spoof_Local_Binary_Pattern(1, 8, norm_image)
        new_img = myLBP.compute_lbp()

        hist = myLBP.createHistogram(new_img)
        return hist


    #viene effettuato lo splitting tra real e fake  e inserite le informazioni in un csv
    def splitting_real_fake(self, fill_csv_real, fill_csv_fake):
        root_dir = 'Data'
        # fill_csv_real = False
        # fill_csv_fake = False

        try:
            os.makedirs(root_dir + '/hist_real')
        except:
            print("La directory seguente è già stata creata: " + root_dir + "/hist_real")
        try:
            os.makedirs(root_dir + '/hist_fake')
        except:
            print("La directory seguente è già stata creata: " + root_dir + "/hist_fake")

        current_real = '/Real'
        current_fake = '/Fake'

        # Qui andiamo ad inserire histogram in csv per ogni immagine Real
        if fill_csv_real == True:
            src_real = "Data" + current_real

            allFileNames = os.listdir(src_real)

            filesName = np.array(allFileNames)

            filesName = [src_real + '/' + name for name in filesName.tolist()]

            for name in filesName:
                hist_real = ReplayAttackSplitting().convert_image_to_hist(name)
                # print(len(hist_real))
                ReplayAttackSplitting().writeCsv(hist_real, 0)

            # shutil.copy(name, "Data/hist_real/")

        # Qui andiamo ad inserire histogram in csv per ogni immagine Fake
        if fill_csv_fake == True:
            src_fake = "Data" + current_fake

            allFileNames = os.listdir(src_fake)

            filesName = np.array(allFileNames)

            filesName = [src_fake + '/' + name for name in filesName.tolist()]

            for name in filesName:
                hist_real = ReplayAttackSplitting().convert_image_to_hist(name)
                ReplayAttackSplitting().writeCsv(hist_real, 1)

# #FUNZIONE TEMPORANEA INIZIO:
# def count_print_row(filecsv):
#
#
#     with open(filecsv, 'r') as csvFile:
#         reader = csv.reader(csvFile, delimiter=';')
#         for row in reader:
#             #print(" ".join(row))
#             print("La lunghezza è: "+ str(len(row)))
#             #print("\n\n\n")
#
#     csvFile.close()
#
# #FINE







def main():



    nameFileCsv = 'histogram.csv'
    ReplayAttackSplitting(nameFileCsv).splitting_real_fake(False,False)




if __name__ == '__main__':
    main()


