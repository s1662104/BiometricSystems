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
    def createDataSetEyeBlink(self, real, fake, nameFileCsv):
        root_dir = 'Data/EyeBlink/'
        current_real = 'Real/'
        current_fake = 'Fake/'
        #Real part
        #In src_real, abbiamo la directory con i video genuini (Real)
        src_real = root_dir + current_real

        #Andiamo a leggere questa directory, ottenenendo una lista dei file (in questo caso video)
        realFileNames = os.listdir(src_real)

        #andiamo a creare una lista dove per ogni elemento, abbiamo il path compreso di file
        # per esempio /Data/EyeBlink/Real/video1.mp4
        realFileNames = [src_real + name for name in realFileNames]

        #Conteggiamo il numero di video reali.
        print('REAL')
        print('Total video Real: ', len(realFileNames))

        #Fake Part
        #In src_fake, abbiamo la directory con i video non genuini (Fake)
        src_fake = root_dir + current_fake

        # Andiamo a leggere questa directory, ottenenendo una lista dei file (in questo caso video)
        fakeFileNames = os.listdir(src_fake)

        # andiamo a creare una lista dove per ogni elemento, abbiamo il path compreso di file
        # per esempio /Data/EyeBlink/Fake/video1.mp4
        fakeFileNames = [src_fake + name for name in fakeFileNames]

        # Conteggiamo il numero di video non reali.
        print('FAKE')
        print('Total video Fake: ', len(fakeFileNames))

        #Qui, andiamo a realizzare una lista dei video che già sono stati analizzati e presenti quindi nel file csv e
        # non devono essere, quindi, di nuovo analizzati. Visto che sono in totale 1897 video e considerando il processo
        # molto lungo, potrebbero capitare imprevisti e sarebbe frustrante ricominciare tutto da capo.
        black_list = []
        try:
            black_list = self.readEyeBlinkCsv(nameFileCsv, 0)
        except Exception as e:
            print(str(e))
        print(black_list)
        print()
        # TODO questi due IF sono uguali, cambia solo un valore. Si potrebbe tutto scrivere in un paio di righe
        # TODO RESPONSE Ho lasciato così, lo stavo provando a modificare ma potrebbe poi non funzionare, non sapendo poi cosa mettere in comune.
        #  Abbiamo due for che prendono dati diversi, lascerei così.

        # in questo caso vengono analizzati i video real se il flag real == True.
        if real:
            # per ogni video reale
            for name in realFileNames:
                #controlliamo se lo abbiamo già analizzato e in caso positivo saltiamo al prossimo video.
                if name in black_list:
                    print(len(black_list))
                    continue
                # appendiamo alla lista il nome del video(compreso di path), il valore 1 perché genuino.
                list = []
                list.append(name)
                list.append(1)
                print(name)
                #chiamiamo la funzione per analizzare il video che ci ritorna true o false
                var = EyeBlink.EyeBlink(name).eyeBlinkStart()
                if var:
                    val = 1
                else:
                    val = 0
                # scriviamo il tutto su file csv.
                self.writeEyeBlinkCsv(nameFileCsv, list, val)
        # in questo caso vengono analizzati i video fake se il flag fake == True.
        if fake:
            # per ogni video fake
            for name in fakeFileNames:
                # controlliamo se lo abbiamo già analizzato e in caso positivo saltiamo al prossimo video.
                if name in black_list:
                    print(len(black_list))
                    continue
                # appendiamo alla lista il nome del video(compreso di path), il valore 0 perché non genuino e il valore ritornato
                # dalla funzione eyeBlinkStart().
                list = []
                list.append(name)
                list.append(0)
                print(name)
                # chiamiamo la funzione per analizzare il video che ci ritorna true o false
                var = EyeBlink.EyeBlink(name).eyeBlinkStart()

                if var:
                    val = 1
                else:
                    val = 0
                # scriviamo il tutto su file csv.
                self.writeEyeBlinkCsv(nameFileCsv, list, val)

    # Qui scriviamo nella lista la colonna data dal valore val.
    def readEyeBlinkCsv(self, nameFileCsv, val):
        list2 = []

        with open(nameFileCsv, 'r') as f:
            csv_reader = csv.reader(f, delimiter=';')
            for row in csv_reader:
                list2.append(row[val])
        return list2

    # vengono letti i video dalle rispettive directory EyeBlink e valutati inserendo le informazioni nel
    # csv, in questo caso viene utilizzato Eye-blink con threshold fisso variabile
    def createDataSetEyeBlinkFixedTh(self, nameFileCsv, real, fake):
        root_dir = 'Data/EyeBlink/'
        current_real = 'Real/'
        current_fake = 'Fake/'
        # Real part
        # In src_real, abbiamo la directory con i video genuini (Real)
        src_real = root_dir + current_real

        # Andiamo a leggere questa directory, ottenenendo una lista dei file (in questo caso video)
        realFileNames = os.listdir(src_real)

        # andiamo a creare una lista dove per ogni elemento, abbiamo il path compreso di file
        # per esempio /Data/EyeBlink/Real/video1.mp4
        realFileNames = [src_real + name for name in realFileNames]

        # Conteggiamo il numero di video reali.
        print('REAL')
        print('Total video Real: ', len(realFileNames))

        # Fake Part
        # In src_fake, abbiamo la directory con i video non genuini (Fake)
        src_fake = root_dir + current_fake

        # Andiamo a leggere questa directory, ottenenendo una lista dei file (in questo caso video)
        fakeFileNames = os.listdir(src_fake)

        # andiamo a creare una lista dove per ogni elemento, abbiamo il path compreso di file
        # per esempio /Data/EyeBlink/Fake/video1.mp4
        fakeFileNames = [src_fake + name for name in fakeFileNames]

        # Conteggiamo il numero di video non reali.
        print('FAKE')
        print('Total video Fake: ', len(fakeFileNames))
        # TODO questi due if sono uguali, cambia solo un valore. Si potrebbe tutto scrivere nella meta' delle righe
        # TODO response: è vero ma abbiamo due for diversi, dovresti uscire dal for ogni volta per scrivere all'interno del csv.
        # in questo caso vengono analizzati i video real se il flag real == True.
        if real:
            for name in realFileNames:
                # appendiamo alla lista il nome del video(compreso di path), il valore 1 perché genuino.
                list = []
                list.append(name)
                list.append(1)
                print(name)
                # chiamiamo la funzione per analizzare il video che ci ritorna una lista di valori relativi ai threshold
                ear_th = EyeBlink.EyeBlink(name).eyeBlinkStartThFixed()
                print(ear_th)
                # scriviamo il tutto su filecsv
                self.writeEyeBlinkCsv(nameFileCsv, list, ear_th)
        # in questo caso vengono analizzati i video non real se il flag fake == False.
        if fake:
            for name in fakeFileNames:
                list = []
                list.append(name)
                list.append(0)
                print(name)
                # chiamiamo la funzione per analizzare il video che ci ritorna una lista di valori relativi ai threshold
                ear_th = EyeBlink.EyeBlink(name).eyeBlinkStartThFixed()
                print(ear_th)
                # scriviamo il tutto su filecsv
                self.writeEyeBlinkCsv(nameFileCsv, list, ear_th)

    # qui viene effettuata l'evaluation per eyeBlink con threshold adattivo
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
    def evaluationFixedThreshold(self,nameFileCsv):
         # lettura dei dati dal csv.
         data = pd.read_csv(nameFileCsv, sep=';', header = None)

         # in y_test avremo una lista dei valori: 1 se il video è reale, 0 se il video non è reale,
         # mentre per y_test_score abbiamo la lista di array (ogni array è visto come una stringa)
         # dei risultati relativi ai threshold per ogni singolo video.
         y_test, y_test_score = data.iloc[:, 1], data.iloc[:, -1]

         # Dichiarazione di liste di false acceptance, false rejection, spoof false acceptance rate,
         # genuine acceptante rate e thresholds.
         fa=[]
         fr=[]
         sfar = []
         frr = []

         # serve per la roc curve, gar
         gar = []
         thresholds = []

         #inizializziamo la lista dei false rejection e dei false acceptance, con tutti
         #i valori uguali a 0. Inizializziamo anche la lista dei thresholds che andranno da
         #0.10 a 0.29.
         for threshold in np.arange(0.10, 0.30, 0.01):
             fr.append(0)
             fa.append(0)
             thresholds.append(round(threshold,2))

         #per tutta la lista di y_test
         for i in range(len(y_test)):

            #converte una stringa in una lista
            array = ast.literal_eval(y_test_score[i])

            #Nel caso in cui abbiamo un video fake (non genuino), se abbiamo nell'array relativo a quel video un valore
            #diverso da 0 incrementiamo il false acceptance relativo a quel threshold.
            if y_test[i] == 0:
                for num in range(len(fa)):
                    if array[num] == 1:
                        fa[num] += 1

            #Nel caso in cui abbiamo un video real,se abbiamo nell'array relativo a quel video un valore
            #diverso da 1 incrementiamo il false acceptance relativo a quel threshold.
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

         # vengono calcolati spoof false acceptance rate, false rejection rate e genuine acceptance rate.
         for i in range(len(fa)):
            sfar.append(fa[i] / num_fake)
            frr.append(fr[i] / num_real)
            # per la roc curve mi serve gar
            gar.append(1 - frr[i])

         print("SFAR: ", sfar)
         print("FRR: ", frr)


         #calcolo di equal error rate
         eer_1 = np.array(sfar)[np.nanargmin(np.absolute((np.array(frr) - np.array(sfar))))]
         eer_2 = np.array(frr)[np.nanargmin(np.absolute((np.array(frr) - np.array(sfar))))]
         eer = (eer_1 + eer_2) / 2
         print("EER:", eer)

         #calcolo del threshold di equal error rate
         eer_threshold = np.array(thresholds)[np.nanargmin(np.absolute((np.array(frr) - np.array(sfar))))]
         print("EER Threshold:", eer_threshold)


         #plotting della roc curve
         plt.plot(sfar, gar)
         plt.plot([0, 1], [0, 1], 'k--')
         plt.ylabel("Genuine Acceptance Rate")
         plt.xlabel("Spoofing False Acceptance Rate")
         plt.title('Receiver Operating Characteristic')
         plt.xlim([0.0, 1.0])
         plt.ylim([0.0, 1.0])
         plt.show()

         ###Analisi dei threshold ottenuti
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

         # trova i valori di sfar e frr corrispondenti al EER threshold.
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
