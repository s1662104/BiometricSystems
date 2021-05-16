import csv
import os

import cv2
import numpy as np
import pandas as pd
import MicroTexture
import EyeBlink
import AntiSpoofingTrainingEvaluation as evaluation


class AntiSpoofingOverallEvaluation():
    def __init__(self, nomeFileCsv):
        self.nomeFileCsv = nomeFileCsv


    # va a scrivere su un file csv: nameVid:=nome del video, val:= valore aspettato, val_pred_eyeblink:= valore predetto dall'algoritmo di eyeblink,
    # val_pred_replayattack := valore predetto dall'algoritmo di microtexture, val_pred_final := valore finale predetto.
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


    def evaluationCascade(self):
        current_real = 'Genuine'
        current_fake_replayattack = 'ReplayAttack'
        current_fake_eyeblink = 'EyeBlinkFake'

        src_real = "EvaluationDataset/" + current_real
        allFileNames = os.listdir(src_real)
        filesName = np.array(allFileNames)
        filesNameReal = [src_real + '/' + name for name in filesName.tolist()]

        for name in filesNameReal:
            varEyeBlink = EyeBlink.EyeBlink(name).eyeBlinkStart()
            if varEyeBlink == 1:
                varMicroTexture = MicroTexture.MicroTexture(nameFileCsv='histogram.csv').microTextureVideo(name)
                if varMicroTexture == True:
                    self.writeCsv(name, 1, 1, 1, 1)
                else:
                    self.writeCsv(name, 1, 1, 0 , 0)
            else:
                self.writeCsv(name, 1 , 0 , 0 , 0)

        src_fake = "EvaluationDataset/" + current_fake_replayattack
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesNameFake:
            varEyeBlink = EyeBlink.EyeBlink(name).eyeBlinkStart()
            if varEyeBlink == 1:
                varMicroTexture = MicroTexture.MicroTexture(nameFileCsv='histogram.csv').microTextureVideo(name)
                if varMicroTexture == True:
                    self.writeCsv(name, 0, 1, 1, 1)
                else:
                    self.writeCsv(name, 0, 1, 0 , 0)
            else:
                self.writeCsv(name, 0 , 0 , 0 , 0)

        src_fake = "EvaluationDataset/" + current_fake_eyeblink
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesNameFake:
            varEyeBlink = EyeBlink.EyeBlink(name).eyeBlinkStart()
            if varEyeBlink == 1:
                varMicroTexture = MicroTexture.MicroTexture(nameFileCsv='histogram.csv').microTextureVideo(name)
                if varMicroTexture == True:
                    self.writeCsv(name, 0, 1, 1, 1)
                else:
                    self.writeCsv(name, 0, 1, 0, 0)
            else:
                self.writeCsv(name, 0, 0, 0, 0)

    def evaluationAND(self):
        current_real = 'Genuine'
        current_fake_replayattack = 'ReplayAttack'
        current_fake_eyeblink = 'EyeBlinkFake'

        src_real = "EvaluationDataset/" + current_real
        allFileNames = os.listdir(src_real)
        filesName = np.array(allFileNames)
        filesNameReal = [src_real + '/' + name for name in filesName.tolist()]

        for name in filesNameReal:
            if ((MicroTexture.MicroTexture(nameFileCsv='histogram.csv').microTextureVideo(name)) and (EyeBlink.EyeBlink(name).eyeBlinkStart() == 1)):
                self.writeCsvAND_OR(name, 1, 1)
            else:
                self.writeCsvAND_OR(name,1,0)


        src_fake = "EvaluationDataset/" + current_fake_replayattack
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesNameFake:
            if ((MicroTexture.MicroTexture(nameFileCsv='histogram.csv').microTextureVideo(name)) and (EyeBlink.EyeBlink(name).eyeBlinkStart() == 1)):
                self.writeCsvAND_OR(name, 0, 1)
            else:
                self.writeCsvAND_OR(name,0,0)

        src_fake = "EvaluationDataset/" + current_fake_eyeblink
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesNameFake:
            if ((MicroTexture.MicroTexture(nameFileCsv='histogram.csv').microTextureVideo(name)) and (EyeBlink.EyeBlink(name).eyeBlinkStart() == 1)):
                self.writeCsvAND_OR(name, 0, 1)
            else:
                self.writeCsvAND_OR(name,0,0)


    def evaluationOR(self):
        current_real = 'Genuine'
        current_fake_replayattack = 'ReplayAttack'
        current_fake_eyeblink = 'EyeBlinkFake'

        src_real = "EvaluationDataset/" + current_real
        allFileNames = os.listdir(src_real)
        filesName = np.array(allFileNames)
        filesNameReal = [src_real + '/' + name for name in filesName.tolist()]

        for name in filesNameReal:
            if ((MicroTexture.MicroTexture(nameFileCsv='histogram.csv').microTextureVideo(name)) or (EyeBlink.EyeBlink(name).eyeBlinkStart() == 1)):
                self.writeCsvAND_OR(name, 1, 1)
            else:
                self.writeCsvAND_OR(name,1,0)


        src_fake = "EvaluationDataset/" + current_fake_replayattack
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesNameFake:
            if ((MicroTexture.MicroTexture(nameFileCsv='histogram.csv').microTextureVideo(name)) or (EyeBlink.EyeBlink(name).eyeBlinkStart() == 1)):
                self.writeCsvAND_OR(name, 0, 1)
            else:
                self.writeCsvAND_OR(name,0,0)

        src_fake = "EvaluationDataset/" + current_fake_eyeblink
        allFileNames = os.listdir(src_fake)
        filesName = np.array(allFileNames)
        filesNameFake = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesNameFake:
            if ((MicroTexture.MicroTexture(nameFileCsv='histogram.csv').microTextureVideo(name)) or (EyeBlink.EyeBlink(name).eyeBlinkStart() == 1)):
                self.writeCsvAND_OR(name, 0, 1)
            else:
                self.writeCsvAND_OR(name,0,0)










    def antispoofingEvaluation(self):
        data = pd.read_csv(self.nomeFileCsv, sep=';', header=None)
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




def main():
    nameFileCsv = 'antispoofingCascade.csv'
    AntiSpoofingOverallEvaluation(nameFileCsv).evaluationCascade()
    # nameFileCsv = 'antispoofingAND.csv'
    # AntiSpoofingOverallEvaluation(nameFileCsv).evaluationAND()
    # nameFileCsv = 'antispoofingOR.csv'
    # AntiSpoofingOverallEvaluation(nameFileCsv).evaluationOR()
    print("CASCADE")
    nameFileCsv = 'antispoofingCascade.csv'
    AntiSpoofingOverallEvaluation(nameFileCsv).antispoofingEvaluation()
    # print("AND")
    # nameFileCsv = 'antispoofingAND.csv'
    # AntiSpoofingOverallEvaluation(nameFileCsv).antispoofingEvaluation()
    # print("OR")
    # nameFileCsv = 'antispoofingOR.csv'
    # AntiSpoofingOverallEvaluation(nameFileCsv).antispoofingEvaluation()





if __name__ == '__main__':
    main()