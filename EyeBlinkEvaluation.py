#Qui viene effettuata l'evaluation di Eyeblink

import csv
import os
from shutil import which

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve

import EyeBlink


import AntiSpoofingTrainingEvaluation as evaluation


class EyeBlinkEvaluation:
    def __init__(self):
        pass


    #viene creato il file csv e scritti i valori al suo interno
    def writeEyeBlinkCsv(self, eyeblink, val):
        print(eyeblink)
        list = []
        for val_array in eyeblink:
            list.append(val_array)
        list.append(val)

        with open('eyeblink.csv', 'a+') as cvsfile:
            writer = csv.writer(cvsfile, delimiter=';')
            writer.writerow(list)


            cvsfile.close()


     #vengono letti i video dalle rispettive directory EyeBlink e valutati inserendo le informazioni nel csv.
    def createDataSetEyeBlink(self, real, fake):
        root_dir = 'Data/EyeBlink/'
        current_real = 'Real/'
        current_fake = 'Fake/'

        ###Real Part
        src_real = root_dir + current_real

        realFileNames = os.listdir(src_real)

        realFileNames = [src_real + name for name in realFileNames]

        print('REAL')
        print('Total video Real: ', len(realFileNames))

        ###Fake Part
        src_fake = root_dir + current_fake

        fakeFileNames = os.listdir(src_fake)

        fakeFileNames = [src_fake + name for name in fakeFileNames]

        print('FAKE')
        print('Total video Fake: ', len(fakeFileNames))
        black_list = []
        try:
            black_list = self.readEyeBlinkCsv(0)
        except Exception as e:
            print(str(e))
        print(black_list)
        print()
        if real == True:
            for name in realFileNames:
                if name in black_list:
                    print(len(black_list))
                    continue
                else:
                    list = []
                    list.append(name)
                    list.append(1)
                    print(name)
                    var = EyeBlink.EyeBlink(name).eyeBlinkStart()
                    if var:
                        val = 1
                    else:
                        val = 0

                    self.writeEyeBlinkCsv(list, val)
        if fake == True:
            for name in fakeFileNames:
                if name in black_list:
                    print(len(black_list))
                    continue
                list = []
                list.append(name)
                list.append(0)
                print(name)

                var = EyeBlink.EyeBlink(name).eyeBlinkStart()

                if var:
                    val = 1
                else:
                    val = 0

                self.writeEyeBlinkCsv(list, val)



    def readEyeBlinkCsv(self, val):
        list2 = []
        with open("eyeblink.csv", 'r') as f:
            csv_reader = csv.reader(f, delimiter=';')
            for row in csv_reader:
                list2.append(row[val])
        return list2


    #qui viene effettuata l'evaluation per eyeBlink
    def evaluation(self,nameFileCsv):
        data = pd.read_csv(nameFileCsv, sep=';', header=None)
        y_test, y_test_score = data.iloc[:, 1], data.iloc[:, -1]
        print("###y_test###")
        print(y_test)
        print("##############")
        print("###y_score###")
        print(y_test_score)
        print("##############")
        print("###SPOOFING SCENARIO###")
        FRR, SFAR = evaluation.spoofing_scenario(y_test, y_test_score, index=0)
        print("FRR", FRR)
        print("SFAR", SFAR)
        print("##############")
        

        print("ROC CURVE:")
        evaluation.plot_roc_curve(y_test, y_test_score)

        print("DET CURVE")
        evaluation.plot_det_curve(y_test, y_test_score)


def main():
    #EyeBlinkEvaluation().createDataSetEyeBlink(False, False)
    EyeBlinkEvaluation().evaluation(nameFileCsv='eyeblink.csv')










if __name__ == '__main__':
    main()


