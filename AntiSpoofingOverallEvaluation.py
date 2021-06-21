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

    # TODO commentare: scrivere almeno cosa sono i valori in input
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

    # TODO commentare ogni passaggio
    def evaluationCascade(self):
        current_real = 'Genuine'
        current_fake_replayattack = 'ReplayAttack'
        current_fake_eyeblink = 'EyeBlinkFake'

        src_real = "EvaluationDataset/" + current_real
        allFileNames = os.listdir(src_real)
        filesName = np.array(allFileNames)
        filesNameReal = [src_real + '/' + name for name in filesName.tolist()]

        # TODO SPIEGARE COSA SONO QUESTI VALORI (1,0) ED EVENTUALMENTE SCRIVERE ANCHE MEGLIO LA FUNZIONE VISTO
        # CHE CAMBIANO SOLO I VALORI 1 E 0. SI POTREBBE SCRIVEREE TUTTO IN POCHE RIGHE DI CODICE
        for name in filesNameReal:
            varEyeBlink = EyeBlink(name).eyeBlinkStart()
            if varEyeBlink == 1:
                varMicroTexture = MicroTexture().microTextureVideo(name)
                if varMicroTexture:
                    self.writeCsv(name, 1, 1, 1, 1)
                else:
                    self.writeCsv(name, 1, 1, 0, 0)
            else:
                self.writeCsv(name, 1, 0, 0, 0)

        src_fake = "EvaluationDataset/" + current_fake_replayattack
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesNameFake:
            varEyeBlink = EyeBlink(name).eyeBlinkStart()
            if varEyeBlink == 1:
                varMicroTexture = MicroTexture().microTextureVideo(name)
                if varMicroTexture:
                    self.writeCsv(name, 0, 1, 1, 1)
                else:
                    self.writeCsv(name, 0, 1, 0, 0)
            else:
                self.writeCsv(name, 0, 0, 0, 0)

        src_fake = "EvaluationDataset/" + current_fake_eyeblink
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesNameFake:
            varEyeBlink = EyeBlink(name).eyeBlinkStart()
            if varEyeBlink == 1:
                varMicroTexture = MicroTexture().microTextureVideo(name)
                if varMicroTexture:
                    self.writeCsv(name, 0, 1, 1, 1)
                else:
                    self.writeCsv(name, 0, 1, 0, 0)
            else:
                self.writeCsv(name, 0, 0, 0, 0)

    # TODO COMMENTARE OGNI PASSAGGIO. SPIEGARE LA DIFFERENZA TRA I SINGOLI FOR
    def evaluationAND(self):
        current_real = 'Genuine'
        current_fake_replayattack = 'ReplayAttack'
        current_fake_eyeblink = 'EyeBlinkFake'

        src_real = "EvaluationDataset/" + current_real
        allFileNames = os.listdir(src_real)
        filesName = np.array(allFileNames)
        filesNameReal = [src_real + '/' + name for name in filesName.tolist()]

        for name in filesNameReal:
            if ((MicroTexture().microTextureVideo(name)) and (
                    EyeBlink(name).eyeBlinkStart() == 1)):
                self.writeCsvAND_OR(name, 1, 1)
            else:
                self.writeCsvAND_OR(name, 1, 0)

        src_fake = "EvaluationDataset/" + current_fake_replayattack
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesNameFake:
            if ((MicroTexture().microTextureVideo(name)) and (
                    EyeBlink(name).eyeBlinkStart() == 1)):
                self.writeCsvAND_OR(name, 0, 1)
            else:
                self.writeCsvAND_OR(name, 0, 0)

        src_fake = "EvaluationDataset/" + current_fake_eyeblink
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesNameFake:
            if ((MicroTexture().microTextureVideo(name)) and (
                    EyeBlink(name).eyeBlinkStart() == 1)):
                self.writeCsvAND_OR(name, 0, 1)
            else:
                self.writeCsvAND_OR(name, 0, 0)

    # TODO COMMENTARE OGNI PASSAGGIO. SPIEGARE LA DIFFERENZA TRA I SINGOLI FOR
    def evaluationOR(self):
        current_real = 'Genuine'
        current_fake_replayattack = 'ReplayAttack'
        current_fake_eyeblink = 'EyeBlinkFake'

        src_real = "EvaluationDataset/" + current_real
        allFileNames = os.listdir(src_real)
        filesName = np.array(allFileNames)
        filesNameReal = [src_real + '/' + name for name in filesName.tolist()]

        for name in filesNameReal:
            if ((MicroTexture().microTextureVideo(name)) or (
                    EyeBlink(name).eyeBlinkStart() == 1)):
                self.writeCsvAND_OR(name, 1, 1)
            else:
                self.writeCsvAND_OR(name, 1, 0)

        src_fake = "EvaluationDataset/" + current_fake_replayattack
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesNameFake:
            if ((MicroTexture().microTextureVideo(name)) or (
                    EyeBlink(name).eyeBlinkStart() == 1)):
                self.writeCsvAND_OR(name, 0, 1)
            else:
                self.writeCsvAND_OR(name, 0, 0)

        src_fake = "EvaluationDataset/" + current_fake_eyeblink
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesNameFake:
            if ((MicroTexture().microTextureVideo(name)) or (
                    EyeBlink(name).eyeBlinkStart() == 1)):
                self.writeCsvAND_OR(name, 0, 1)
            else:
                self.writeCsvAND_OR(name, 0, 0)

    # TODO COMMENTARE OGNI PASSAGGIO.
    def antispoofingEvaluation(self):
        data = pd.read_csv(self.nomeFileCsv, sep=';', header=None)
        y_test, y_test_score = data.iloc[:, 1], data.iloc[:, -1]
        print("###SPOOFING SCENARIO###")
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
