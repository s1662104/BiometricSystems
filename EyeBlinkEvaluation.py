#Qui viene effettuata l'evaluation di Eyeblink
import ast
import csv
import os
from shutil import which

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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

        # with open('eyeblink.csv', 'a+') as cvsfile:
        #     writer = csv.writer(cvsfile, delimiter=';')
        #     writer.writerow(list)

        with open('eyeblinkFixedTh.csv', 'a+') as cvsfile:
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
        ## FIXED TH
        fr = []
        fa = []
        ##
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
        # with open("eyeblink.csv", 'r') as f:
        #     csv_reader = csv.reader(f, delimiter=';')
        #     for row in csv_reader:
        #         list2.append(row[val])

        with open("eyeblinkFixedTh.csv", 'r') as f:
           csv_reader = csv.reader(f, delimiter=';')
           for row in csv_reader:
               list2.append(row[val])
        return list2



    def createDataSetEyeBlinkFixedTh(self, real, fake):
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

        # ## FIXED TH
        # fr = []
        # fa = []
        # for threshold in np.arange(0.10, 0.30, 0.01):
        #     fr.append(0)
        #     fa.append(0)
        # ##
        if real == True:
            for name in realFileNames:
                list = []
                list.append(name)
                list.append(1)
                print(name)
                ear_th = EyeBlink.EyeBlink(name).eyeBlinkStartThFixed()
                print(ear_th)
                #count = 0
                # for val in ear_th:
                #     if val == 0:
                #         fr[count] += 1
                #     count += 1


                    # if var:
                    #     val = 1
                    # else:
                    #     val = 0
                    #
                self.writeEyeBlinkCsv(list, ear_th)
        if fake == True:
            for name in fakeFileNames:
                list = []
                list.append(name)
                list.append(0)
                print(name)
                ear_th = EyeBlink.EyeBlink(name).eyeBlinkStartThFixed()
                print(ear_th)
                # count = 0
                # for val in ear_th:
                #     if val == 0:
                #         fr[count] += 1
                #     count += 1

                # if var:
                #     val = 1
                # else:
                #     val = 0
                #
                self.writeEyeBlinkCsv(list, ear_th)

                    # if var:
                #     val = 1
                # else:
                #     val = 0
                #
                # self.writeEyeBlinkCsv(list, val)


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
        

        # print("ROC CURVE:")
        # evaluation.plot_roc_curve(y_test, y_test_score)

        # print("DET CURVE")
        # evaluation.plot_det_curve(y_test, y_test_score)
    # def readEyeBlinkFixedThCsv(self,nameCsv,val):
    #     list2 = []
    #     with open(nameCsv, 'r') as f:
    #         csv_reader = csv.reader(f, delimiter=';')
    #         for row in csv_reader:
    #             list2.append(row[val])
    #     return list2
    def evaluationFixedThreshold(self,nameFileCsv):
         data = pd.read_csv(nameFileCsv, sep=';', header = None)
         y_test, y_test_score = data.iloc[:, 1], data.iloc[:, -1]
         fa=[]
         fr=[]
         sfar = []
         frr = []
         #serve per la roc curve
         gar = []
         thresholds = []
         for threshold in np.arange(0.10, 0.30, 0.01):
             fr.append(0)
             fa.append(0)
             thresholds.append(round(threshold,2))

         ##Fake
         for i in range(len(y_test)):
            #converte una stringa in una lista
            array = ast.literal_eval(y_test_score[i])
            if y_test[i] == 0:
                for num in range(len(fa)):
                    if (array[num] == 1):
                        fa[num] += 1
         ##Real
            elif y_test[i] == 1:
                for num in range(len(fr)):
                    if(array[num] == 0):
                        fr[num] += 1

         print("FA: ",fa)
         print("FR: ",fr)

         #conta i video reali e non
         num_real = 0
         num_fake = 0
         for i in range(len(y_test)):
             if y_test[i] == 1:
                 num_real += 1
             else:
                 num_fake += 1
         for i in range(len(fa)):
             sfar.append(fa[i]/num_fake)
             frr.append(fr[i]/num_real)
             #per la roc curve mi serve gar
             gar.append(1-frr[i])

         print("SFAR: ", sfar)
         print("FRR: ", frr)

         eer_1 = np.array(sfar)[np.nanargmin(np.absolute((np.array(frr) - np.array(sfar))))]
         eer_2 = np.array(frr)[np.nanargmin(np.absolute((np.array(frr) - np.array(sfar))))]
         eer = (eer_1 + eer_2) / 2
         print("EER:", eer)

         eer_threshold = np.array(thresholds)[np.nanargmin(np.absolute((np.array(frr) - np.array(sfar))))]
         print("EER Threshold:", eer_threshold)

         plt.plot(sfar, gar)
         plt.plot([0, 1], [0, 1], 'k--')
         plt.ylabel("Genuine Acceptance Rate")
         plt.xlabel("False Acceptance Rate")
         plt.title('Receiver Operating Characteristic')
         plt.xlim([0.0, 1.0])
         plt.ylim([0.0, 1.0])
         plt.show()


         ###Analisi threshold ottenuti
         print()
         print("ANALISI THRESHOLD OTTENUTI")
         print()
         #trovo il valore minimo e ritorno l'index dei rispettivi SFAR e FRR
         min_value_sfar = min(sfar)
         min_index_sfar = sfar.index(min_value_sfar)
         min_value_frr = min(frr)
         min_index_frr = frr.index(min_value_frr)

         #trova il frr equivalente al più basso sfar

         print("Minimo SFAR: ", min_value_sfar)
         print("FRR corrispondente al minimo SFAR: ", frr[min_index_sfar])
         print("Threshold corrispondende: ", thresholds[min_index_sfar])
         print()
         print("Minimo FRR: ", min_value_frr)
         print("SFAR corrispondente al minimo FRR", sfar[min_index_frr])
         print("Threshold corrispondende: ", thresholds[min_index_frr])


         index_eer = thresholds.index(eer_threshold)
         print()
         print("Valori di SFAR e FRR in corrispondenza del threshold di EER")
         print("SFAR riferito a EER: ", sfar[index_eer])
         print("FRR riferito a EER: ", frr[index_eer])

         return








def main():
    EyeBlinkEvaluation().evaluationFixedThreshold(nameFileCsv='eyeblinkFixedTh.csv')
    #EyeBlinkEvaluation().createDataSetEyeBlinkFixedTh(False, False)
    #EyeBlinkEvaluation().evaluation(nameFileCsv='eyeblinkFixedTh.csv')










if __name__ == '__main__':
    main()


