# Qui vengono realizzati i due dataset, uno per replayAttack e uno per eyeBlink, prendendo i video da ROSE-YoutubeFace


import cv2
import os
import shutil

import Main


# vengono creati i due dataset ReplayAttack e eyeBlink. Copiando i frame per ReplayAttack in Fake e Real; e copiando
# i video per eyeBlink.
def createDataSet(input, val, name, replayAttack, eyeBlink):
    if replayAttack:
        cap = cv2.VideoCapture(input)
        pathReal = 'Data/ReplayAttack/Real/'
        pathFake = 'Data/ReplayAttack/Fake/'
        counter = 0

        while True:

            ret, frame = cap.read()

            vis = frame.copy()

            if vis is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop = Main.detect_face(gray, vis)

            if crop is not None:
                counter += 1
                if val == 'Real':
                    if not os.path.exists(pathReal):
                        os.makedirs(pathReal)
                    try:
                        cv2.imwrite(pathReal + name + "_{}.jpg".format(counter), crop)
                    except Exception as e:
                        print(str(e))
                elif val == 'Fake':
                    if not os.path.exists(pathFake):
                        os.makedirs(pathFake)
                    try:
                        cv2.imwrite(pathFake + name + "_{}.jpg".format(counter), crop)
                    except Exception as e:
                        print(str(e))
            break
        cap.release()
        cv2.destroyAllWindows()

    elif eyeBlink:
        pathReal = 'Data/EyeBlink/Real/'
        pathFake = 'Data/EyeBlink/Fake/'
        if val == 'Real':
            if not os.path.exists(pathReal):
                os.makedirs(pathReal)
            try:
                shutil.copy(input, pathReal)
            except Exception as e:
                print(str(e))
        elif val == 'Fake':
            if not os.path.exists(pathFake):
                os.makedirs(pathFake)
            try:
                shutil.copy(input, pathFake)
            except Exception as e:
                print(str(e))


# Qui i video in "Rose - Youtube Face" vengono suddivisi tra Real e Fake
# e viene richiamata la funzione createDataSet per creare
# i rispettivi dataset per ReplayAttack e Eyeblink.
class Database:

    def __init__(self, index):
        # index è '0' Database ROSE per Replay Attack
        # index è '1' Database ROSE per EyeBlink

        self.data = []
        self.target = []
        if index == 0:

            root = 'ROSE - Youtube Face'

            for path, subdirs, files in os.walk(root):
                for name in files:
                    if not name.startswith(('Mc', 'Mf', 'Mu', 'Ml')):
                        if name.startswith('G'):
                            input = os.path.join(path, name)
                            print(input)
                            createDataSet(input, 'Real', name, True, False)
                        else:
                            input = os.path.join(path, name)
                            print(input)
                            createDataSet(input, 'Fake', name, True, False)
        elif index == 1:

            root = 'ROSE - Youtube Face'
            for path, subdirs, files in os.walk(root):
                for name in files:
                    if not name.startswith(('Mc', 'Mu', 'Ml')):
                        if name.startswith('G'):
                            input = os.path.join(path, name)
                            print(input)
                            createDataSet(input, 'Real', name, False, True)
                        elif name.startswith(('Mf', 'Ps', 'Pq')):
                            input = os.path.join(path, name)
                            print(input)
                            createDataSet(input, 'Fake', name, False, True)


if __name__ == '__main__':
    # Database(0)
    Database(1)
