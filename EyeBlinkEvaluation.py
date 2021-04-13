import csv
import os
from shutil import which

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve

import Antispoofing


import AntiSpoofingTrainingEvaluation as evaluation


class EyeBlinkEvaluation:
    def __init__(self):
        pass

    def writeEyeBlinkCsv(self, eyeblink, val):
        print(eyeblink)
        list = []
        for val_array in eyeblink:
            list.append(val_array)
        list.append(val)

        with open('eyeblink.csv', 'a+') as cvsfile:
            writer = csv.writer(cvsfile, delimiter=';')
            writer.writerow(list)

            # cvsfile.write(str(val))
            # cvsfile.write('\n')
            cvsfile.close()

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
                    var = Antispoofing.EyeBlink(name).eyeBlinkStart()
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

                var = Antispoofing.EyeBlink(name).eyeBlinkStart()

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

    def readEyeBlinkBlackCsv(self, val):
        list2 = []
        with open("black_eyeblink.csv", 'r') as f:
            csv_reader = csv.reader(f, delimiter=';')
            for row in csv_reader:
                list2.append(row[val])
        return list2


def main():
    #EyeBlinkEvaluation().createDataSetEyeBlink(False, False)
    data = pd.read_csv("eyeblink.csv", sep=';', header=None)
    y_test, y_test_score = data.iloc[:, 1], data.iloc[:, -1]
    print("###y_test###")
    print(y_test)
    print("##############")
    print("###y_score###")
    print(y_test_score)
    print("##############")
    print("###SPOOFING SCENARIO###")
    FRR, SFAR = evaluation.spoofing_scenario(y_test, y_test_score)
    print("FRR", FRR)
    print("SFAR", SFAR)
    print("##############")
    print("###LICIT SCENARIO###")
    FRR, FAR, HTER = evaluation.licit_scenario(y_test,y_test_score)
    print("FAR",FAR)
    print("FRR",FRR)
    print("HTER", HTER)
    print("##############")
    print("### FAR AND FRR ###")
    FAR, FRR = evaluation.calculate_FAR_FRR(y_test, y_test_score)
    print("FAR", FAR)
    print("FRR", FRR)
    print("##############")

    print("ROC CURVE:")
    evaluation.plot_roc_curve(y_test, y_test_score)
    # ##Parte solo real
    # y_test = data.iloc[0:897, 1]
    # #print(y_test)
    # y_test_score = data.iloc[0:897, -1]
    # #print(y_test_score)
    # print("Real",accuracy_score(y_test, y_test_score))
    #
    # ##Parte solo fake
    # y_test = data.iloc[897:1897, 1]
    # #print(y_test)
    # y_test_score = data.iloc[897:1897, -1]
    # #print(y_test_score)
    # print("Fake",accuracy_score(y_test, y_test_score))
    #FRR,FAR,_ = evaluation.licit_scenario(y_test,y_test_score)

    print("DET CURVE")
    evaluation.plot_det_curve(y_test, y_test_score)
    # _,_,threshold = roc_curve(y_test,y_test_score)
    # evaluation.plot_FAR_FRR_ERR(FAR,FRR,threshold)



def temp_comment():
    data = pd.read_csv("eyeblink.csv", sep=';', header=None)
    black = EyeBlinkEvaluation().readEyeBlinkBlackCsv(0)
    number = 1897
    for i in range(number):
        list = []
        if data.iloc[i, 0] in black:
            print(black)
            print(data.iloc[i, 0])
            continue
        if data.iloc[i, 1] != data.iloc[i, -1]:
            print(data.iloc[i, 0])
            name = data.iloc[i, 0]
            var = Antispoofing.EyeBlink(name).eyeBlinkStart()

            ###Scrive un nuovo csv, questa parte è temporanea e di test
            list.append(data.iloc[i, 0])
            list.append(data.iloc[i, 1])
            if var == True:
                list.append(1)
            elif var == False:
                list.append(0)

            with open('black_eyeblink.csv', 'a+') as cvsfile:
                writer = csv.writer(cvsfile, delimiter=';')
                writer.writerow(list)

                # cvsfile.write(str(val))
                # cvsfile.write('\n')
                cvsfile.close()

            print(name, var)
        else:
            continue
    print(list)
    bData = pd.read_csv("black_eyeblink.csv", sep=';', header=None)
    print(bData)
    y_test = bData.iloc[:,1]
    y_test_score = bData.iloc[:,-1]


    #y_test_score = bData[2].astype(int)
    print(y_test)
    #y_test_score = EyeBlinkEvaluation().readEyeBlinkBlackCsv(-1)
    print(y_test_score)


    #evaluation.plot_roc_curve(y_test, y_test_score)




if __name__ == '__main__':
    main()

    #
    #
    # y_test, y_test_score = data.iloc[:,1], data.iloc[:, -1]
    # #y_test = EyeBlinkEvaluation().readEyeBlinkCsv(1)
    # print(y_test)
    # #y_test_score = EyeBlinkEvaluation().readEyeBlinkCsv(2)
    # print(y_test_score)
    # FRR, SFAR = evaluation.spoofing_scenario(y_test,y_test_score)
    # #evaluation.plot_roc_curve(y_test, y_test_score)
    #
    # ##Parte solo real
    # y_test = data.iloc[0:897,1]
    # print(y_test)
    # y_test_score = data.iloc[0:897,-1]
    # print(y_test_score)
    # y_test = data.iloc[897:1897,1]
    # print(y_test)
    # y_test_score = data.iloc[897:1897,-1]
    # #FRR, SFAR = evaluation.spoofing_scenario(y_test, y_test_score)
    # print(accuracy_score(y_test, y_test_score))
    #
    #
    # #evaluation.plot_roc_curve(y_test, y_test_score)
    # #Antispoofing.EyeBlink(None).eyeBlinkStart()
