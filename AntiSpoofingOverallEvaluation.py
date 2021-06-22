import csv
import os
import numpy as np
import pandas as pd
from MicroTexture import MicroTexture
from EyeBlink import EyeBlink
import AntiSpoofingTrainingEvaluation as evaluation


class AntiSpoofingOverallEvaluation:
    def __init__(self, nomeFileCsv):
        self.nomeFileCsv = nomeFileCsv

    # va a scrivere su un file csv: nameVid:=nome del video, val:= valore aspettato,
    # val_pred_eyeblink:= valore predetto dall'algoritmo di eyeblink,
    # val_pred_replayattack := valore predetto dall'algoritmo di microtexture,
    # val_pred_final := valore finale predetto.
    def writeCsv(self, nameVid, val, val_pred_eyeblink, val_pred_microtexture, val_pred_final):
        print("Video: ", nameVid)
        list = []
        list.append(nameVid)
        list.append(val)
        list.append(val_pred_eyeblink)
        list.append(val_pred_microtexture)
        list.append(val_pred_final)

        with open(self.nomeFileCsv, 'a+') as cvsfile:
            writer = csv.writer(cvsfile, delimiter=';')
            writer.writerow(list)
            cvsfile.close()

    # questa funzione serve per scrivere i CSV di AND e OR, in input abbiamo:
    # nameVid := nome del video
    # val := { 1 se è un video reale , 0 se è un video fake }
    # val_pred_final è la predizione finale che è stata fatta tra i due anti_spoofing combinati con AND or OR
    def writeCsvAND_OR(self, nameVid, val, val_pred_final):
        print("Video: ", nameVid)
        list = []
        list.append(nameVid)
        list.append(val)
        list.append(val_pred_final)

        with open(self.nomeFileCsv, 'a+') as cvsfile:
            writer = csv.writer(cvsfile, delimiter=';')
            writer.writerow(list)
            cvsfile.close()

    # evaluation degli antispoofing in cascata
    def evaluationCascade(self):
        current_real = 'Genuine'
        current_fake_replayattack = 'ReplayAttack'
        current_fake_eyeblink = 'EyeBlinkFake'


        # vengono presi i video genuini dalla relativa directory e messi in una lista.
        src_real = "EvaluationDataset/" + current_real
        allFileNames = os.listdir(src_real)
        filesName = np.array(allFileNames)
        filesNameReal = [src_real + '/' + name for name in filesName.tolist()]



        # per ogni video facciamo partire prima l'antispoofing eyeblink...
        for name in filesNameReal:
            varEyeBlink = EyeBlink(name).eyeBlinkStart()

            # se il risultato è 1 allora significa che il video ha passato il contro di Eyeblink e si procede con MicroTexture
            if varEyeBlink == 1:
                varMicroTexture = MicroTexture().microTextureVideo(name)
                #se anche il video supera il contro microtexture allora scriviamo i nel csv
                # nome del video, che si tratta di un video reale (1), che Eyeblink ha dato True come risultato (1),
                # Microtexture ha dato True come risultato (1) e quindi come ultimo valore (1) perché il video è
                # apparso reale al sistema
                if varMicroTexture:
                    self.writeCsv(name, 1, 1, 1, 1)
                #... eyeblink ok(1), microtexture no (0), quindi il valore finale è uno 0 perché viene visto come non reale
                else:
                    self.writeCsv(name, 1, 1, 0, 0)
                #...infine se già eyeblink ha detto no, quindi non reale, microtexture non viene fatto partire quindi 0
                # e si restituisce che è un fake (0)
            else:
                self.writeCsv(name, 1, 0, 0, 0)

        # vengono presi i video fake replayattack, dalla relativa directory e messi in una lista.
        src_fake = "EvaluationDataset/" + current_fake_replayattack
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        # per ogni video facciamo partire prima l'antispoofing eyeblink...
        for name in filesNameFake:
            varEyeBlink = EyeBlink(name).eyeBlinkStart()

            # se il risultato è 1 allora significa che il video ha passato il contro di Eyeblink e si procede con MicroTexture
            if varEyeBlink == 1:
                varMicroTexture = MicroTexture().microTextureVideo(name)
                # se anche il video supera il contro microtexture allora scriviamo i nel csv
                # nome del video, che si tratta di un video reale (1), che Eyeblink ha dato True come risultato (1),
                # Microtexture ha dato True come risultato (1) e quindi come ultimo valore (1) perché il video è
                # apparso reale al sistema
                if varMicroTexture:
                    self.writeCsv(name, 0, 1, 1, 1)
                # ... eyeblink ok(1), microtexture no (0), quindi il valore finale è uno 0 perché viene visto come non reale
                else:
                    self.writeCsv(name, 0, 1, 0, 0)
                #...infine se già eyeblink ha detto no, quindi non reale, microtexture non viene fatto partire quindi 0
                # e si restituisce che è un fake (0)
            else:
                self.writeCsv(name, 0, 0, 0, 0)

        # vengono presi i video fake eyeblink, dalla relativa directory e messi in una lista.
        src_fake = "EvaluationDataset/" + current_fake_eyeblink
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        # per ogni video facciamo partire prima l'antispoofing eyeblink...
        for name in filesNameFake:
            varEyeBlink = EyeBlink(name).eyeBlinkStart()

            # se il risultato è 1 allora significa che il video ha passato il contro di Eyeblink e si procede con MicroTexture
            if varEyeBlink == 1:
                varMicroTexture = MicroTexture().microTextureVideo(name)
                # se anche il video supera il contro microtexture allora scriviamo i nel csv
                # nome del video, che si tratta di un video reale (1), che Eyeblink ha dato True come risultato (1),
                # Microtexture ha dato True come risultato (1) e quindi come ultimo valore (1) perché il video è
                # apparso reale al sistema
                if varMicroTexture:
                    self.writeCsv(name, 0, 1, 1, 1)
                # ... eyeblink ok(1), microtexture no (0), quindi il valore finale è uno 0 perché viene visto come non reale
                else:
                    self.writeCsv(name, 0, 1, 0, 0)
                # ...infine se già eyeblink ha detto no, quindi non reale, microtexture non viene fatto partire quindi 0
                # e si restituisce che è un fake (0)
            else:
                self.writeCsv(name, 0, 0, 0, 0)

    # evaluation degli antispoofing in AND
    def evaluationAND(self):
        current_real = 'Genuine'
        current_fake_replayattack = 'ReplayAttack'
        current_fake_eyeblink = 'EyeBlinkFake'

        # vengono presi i video genuini dalla relativa directory e messi in una lista.
        src_real = "EvaluationDataset/" + current_real
        allFileNames = os.listdir(src_real)
        filesName = np.array(allFileNames)
        filesNameReal = [src_real + '/' + name for name in filesName.tolist()]

        # per ogni video facciamo partire prima l'antispoofing Microtexture in combinazione con Eyeblink, però bisogna ricordarci
        # che non verranno eseguiti in parallelo ma in seriale, quindi prima viene eseguito MicroTexture e se ha esito
        # positivo allora poi viene eseguito eyeblink.
        for name in filesNameReal:
            if ((MicroTexture().microTextureVideo(name)) and (
                    EyeBlink(name).eyeBlinkStart() == 1)):
                # se entrambi vanno a buon fine andiamo a scrivere che è reale (1)...
                self.writeCsvAND_OR(name, 1, 1)
                # altrimenti scriviamo (0) fake
            else:
                self.writeCsvAND_OR(name, 1, 0)

        # vengono presi i video fake replayattack, dalla relativa directory e messi in una lista.
        src_fake = "EvaluationDataset/" + current_fake_replayattack
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        # per ogni video facciamo partire prima l'antispoofing Microtexture in combinazione (AND) con Eyeblink

        for name in filesNameFake:
            if ((MicroTexture().microTextureVideo(name)) and (
                    EyeBlink(name).eyeBlinkStart() == 1)):
                # se entrambi vanno a buon fine andiamo a scrivere che è reale (1)...
                self.writeCsvAND_OR(name, 0, 1)
                # altrimenti scriviamo (0) fake
            else:
                self.writeCsvAND_OR(name, 0, 0)

        # vengono presi i video fake eyeblink, dalla relativa directory e messi in una lista.
        src_fake = "EvaluationDataset/" + current_fake_eyeblink
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        # per ogni video facciamo partire prima l'antispoofing Microtexture in combinazione (AND) con Eyeblink

        for name in filesNameFake:
            if ((MicroTexture().microTextureVideo(name)) and (
                    EyeBlink(name).eyeBlinkStart() == 1)):
                # se entrambi vanno a buon fine andiamo a scrivere che è reale (1)...
                self.writeCsvAND_OR(name, 0, 1)
                # altrimenti scriviamo (0) fake
            else:
                self.writeCsvAND_OR(name, 0, 0)

    # TODO COMMENTARE OGNI PASSAGGIO. SPIEGARE LA DIFFERENZA TRA I SINGOLI FOR
    def evaluationOR(self):
        current_real = 'Genuine'
        current_fake_replayattack = 'ReplayAttack'
        current_fake_eyeblink = 'EyeBlinkFake'

        # vengono presi i video genuini dalla relativa directory e messi in una lista.
        src_real = "EvaluationDataset/" + current_real
        allFileNames = os.listdir(src_real)
        filesName = np.array(allFileNames)
        filesNameReal = [src_real + '/' + name for name in filesName.tolist()]

        # per ogni video facciamo partire prima l'antispoofing Microtexture in combinazione con Eyeblink, però bisogna ricordarci
        # che non verranno eseguiti in parallelo ma in seriale, quindi prima viene eseguito MicroTexture e se ha esito
        # positivo, eyeblink non viene eseguito, altrimenti sì (essendo un OR, basta anche solo che uno dei due
        # antispoofing abbia un esito positivo)
        for name in filesNameReal:
            if ((MicroTexture().microTextureVideo(name)) or (
                    EyeBlink(name).eyeBlinkStart() == 1)):
                # se anche solo uno dei due va a buon fine 'Real' (1)
                self.writeCsvAND_OR(name, 1, 1)
                # altrimenti se nessuno dei due va a buon fine 'Fake' (0)
            else:
                self.writeCsvAND_OR(name, 1, 0)

        # vengono presi i video fake replayattack, dalla relativa directory e messi in una lista
        src_fake = "EvaluationDataset/" + current_fake_replayattack
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        # per ogni video facciamo partire prima l'antispoofing Microtexture in combinazione (OR) con Eyeblink

        for name in filesNameFake:
            if ((MicroTexture().microTextureVideo(name)) or (
                    EyeBlink(name).eyeBlinkStart() == 1)):
                # se anche solo uno dei due va a buon fine 'Real' (1)
                self.writeCsvAND_OR(name, 0, 1)
                # altrimenti se nessuno dei due va a buon fine 'Fake' (0)
            else:
                self.writeCsvAND_OR(name, 0, 0)

        # vengono presi i video fake eyeblink, dalla relativa directory e messi in una lista.
        src_fake = "EvaluationDataset/" + current_fake_eyeblink
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        # per ogni video facciamo partire prima l'antispoofing Microtexture in combinazione (OR) con Eyeblink

        for name in filesNameFake:
            if ((MicroTexture().microTextureVideo(name)) or (
                    EyeBlink(name).eyeBlinkStart() == 1)):
                # se entrambi vanno a buon fine andiamo a scrivere che è reale (1)...
                self.writeCsvAND_OR(name, 0, 1)
            else:
                # altrimenti scriviamo (0) fake
                self.writeCsvAND_OR(name, 0, 0)

    # queste funzione serve per effettuare la valutazione dell'antispoofing leggendo il corrispettivo CSV
    def antispoofingEvaluation(self):
        # leggi i dati del csv
        data = pd.read_csv(self.nomeFileCsv, sep=';', header=None)
        # in y_test avremo il giusto valore ('Real'(1) 'Fake'(0)) in y_test_score il valore che è stato valutato dai due
        #sistemi
        y_test, y_test_score = data.iloc[:, 1], data.iloc[:, -1]
        print("###SPOOFING SCENARIO###")
        # calcolo di FRR e SFAR.
        FRR, SFAR = evaluation.spoofing_scenario(y_test, y_test_score, index=0)
        print()
        print("FRR", FRR)
        print("SFAR", SFAR)
        print()
        print("#######################")


def main():
    nameFileCsv = 'antispoofingCascade.csv'
    AntiSpoofingOverallEvaluation(nameFileCsv).evaluationCascade()
    nameFileCsv = 'antispoofingAND.csv'
    AntiSpoofingOverallEvaluation(nameFileCsv).evaluationAND()
    nameFileCsv = 'antispoofingOR.csv'
    AntiSpoofingOverallEvaluation(nameFileCsv).evaluationOR()
    print("CASCADE")
    nameFileCsv = 'antispoofingCascade.csv'
    AntiSpoofingOverallEvaluation(nameFileCsv).antispoofingEvaluation()
    print("AND")
    nameFileCsv = 'antispoofingAND.csv'
    AntiSpoofingOverallEvaluation(nameFileCsv).antispoofingEvaluation()
    print("OR")
    nameFileCsv = 'antispoofingOR.csv'
    AntiSpoofingOverallEvaluation(nameFileCsv).antispoofingEvaluation()


if __name__ == '__main__':
    main()
