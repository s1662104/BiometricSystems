# Qui viene effettuata l'evaluation di Eyeblink
import ast
import csv
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import EyeBlink
import AntiSpoofingTrainingEvaluation as evaluation


class EyeBlinkEvaluation:
    def __init__(self):
        pass

    # viene creato il file csv e scritti i valori al suo interno
    def writeEyeBlinkCsv(self, nameFileCsv, eyeblink, val):
        print(eyeblink)
        list = []
        for val_array in eyeblink:
            list.append(val_array)
        list.append(val)

        with open(nameFileCsv, 'a+') as cvsfile:
            writer = csv.writer(cvsfile, delimiter=';')
            writer.writerow(list)

            cvsfile.close()

    # vengono letti i video dalle rispettive directory EyeBlink e valutati inserendo le informazioni nel csv.
    # TODO commentare ogni passaggio
    def createDataSetEyeBlink(self, real, fake, nameFileCsv):
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
            black_list = self.readEyeBlinkCsv(nameFileCsv, 0)
        except Exception as e:
            print(str(e))
        print(black_list)
        print()
        # TODO questi due IF sono uguali, cambia solo un valore. Si potrebbe tutto scrivere in un paio di righe
        if real:
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

                    self.writeEyeBlinkCsv(nameFileCsv, list, val)
        if fake:
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

                self.writeEyeBlinkCsv(nameFileCsv, list, val)

    # TODO commentare
    def readEyeBlinkCsv(self, nameFileCsv, val):
        list2 = []

        with open(nameFileCsv, 'r') as f:
            csv_reader = csv.reader(f, delimiter=';')
            for row in csv_reader:
                list2.append(row[val])
        return list2

    # vengono letti i video dalle rispettive directory EyeBlink e valutati inserendo le informazioni nel
    # csv, in questo caso viene utilizzato Eye-blink con threshold fisso variabile
    # TODO commentare ogni passaggio
    def createDataSetEyeBlinkFixedTh(self, nameFileCsv, real, fake):
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
        # TODO questi due if sono uguali, cambia solo un valore. Si potrebbe tutto scrivere nella meta' delle righe
        if real:
            for name in realFileNames:
                list = []
                list.append(name)
                list.append(1)
                print(name)
                ear_th = EyeBlink.EyeBlink(name).eyeBlinkStartThFixed()
                print(ear_th)
                self.writeEyeBlinkCsv(nameFileCsv, list, ear_th)
        if fake:
            for name in fakeFileNames:
                list = []
                list.append(name)
                list.append(0)
                print(name)
                ear_th = EyeBlink.EyeBlink(name).eyeBlinkStartThFixed()
                print(ear_th)
                self.writeEyeBlinkCsv(nameFileCsv, list, ear_th)

    # qui viene effettuata l'evaluation per eyeBlink
    def evaluation(self, nameFileCsv):
        data = pd.read_csv(nameFileCsv, sep=';', header=None)
        y_test, y_test_score = data.iloc[:, 1], data.iloc[:, -1]
        print("###SPOOFING SCENARIO###")
        print()
        FRR, SFAR = evaluation.spoofing_scenario(y_test, y_test_score, index=0)
        print("FRR", FRR)
        print("SFAR", SFAR)
        print()
        print("########################")

    # calcolo dei risultati con relative rappresentazioni utilizzando i thresholds fissi variabili.
    # TODO commentare ogni passaggio
    def evaluationFixedThreshold(self, nameFileCsv):
        data = pd.read_csv(nameFileCsv, sep=';', header=None)
        y_test, y_test_score = data.iloc[:, 1], data.iloc[:, -1]
        fa = []
        fr = []
        sfar = []
        frr = []
        # serve per la roc curve
        gar = []
        thresholds = []
        for threshold in np.arange(0.10, 0.30, 0.01):
            fr.append(0)
            fa.append(0)
            thresholds.append(round(threshold, 2))

        ##Fake
        for i in range(len(y_test)):
            # converte una stringa in una lista
            array = ast.literal_eval(y_test_score[i])
            if y_test[i] == 0:
                for num in range(len(fa)):
                    if array[num] == 1:
                        fa[num] += 1
            ##Real
            elif y_test[i] == 1:
                for num in range(len(fr)):
                    if array[num] == 0:
                        fr[num] += 1

        print("FA: ", fa)
        print("FR: ", fr)

        # conta i video reali e non
        num_real = 0
        num_fake = 0
        for i in range(len(y_test)):
            if y_test[i] == 1:
                num_real += 1
            else:
                num_fake += 1
        for i in range(len(fa)):
            sfar.append(fa[i] / num_fake)
            frr.append(fr[i] / num_real)
            # per la roc curve mi serve gar
            gar.append(1 - frr[i])

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
        plt.xlabel("Spoofing False Acceptance Rate")
        plt.title('Receiver Operating Characteristic')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.show()

        ###Analisi threshold ottenuti
        print()
        print("ANALISI THRESHOLD OTTENUTI")
        print()
        # trovo il valore minimo e ritorno l'index dei rispettivi SFAR e FRR
        min_value_sfar = min(sfar)
        min_index_sfar = sfar.index(min_value_sfar)
        min_value_frr = min(frr)
        min_index_frr = frr.index(min_value_frr)

        # trova il frr equivalente al più basso sfar

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
    # EyeBlinkEvaluation().createDataSetEyeBlink(False, False, nameFileCsv='eyeblinkAdaptiveTh.csv')
    # EyeBlinkEvaluation().evaluation(nameFileCsv='eyeblinkAdaptiveTh.csv')


if __name__ == '__main__':
    main()
