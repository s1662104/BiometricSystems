


import cv2
import os
import shutil

import Main


# Viene creato il dataset per eyeblink, suddividendo e copiando i video in Real e Fake.
def createDataSet(input, val):

    # i path real e fake
    pathReal = 'Data/EyeBlink/Real/'
    pathFake = 'Data/EyeBlink/Fake/'

    # se il video è di tipo Real creiamo il percorso
    if val == 'Real':
        if not os.path.exists(pathReal):
            os.makedirs(pathReal)
    # copiamo il video nel giusto percorso
        try:
            shutil.copy(input, pathReal)
        except Exception as e:
            print(str(e))
    # se il video è di tipo Fake creiamo il percorso
    elif val == 'Fake':
        if not os.path.exists(pathFake):
            os.makedirs(pathFake)
    # copiamo il video nel giusto percorso
        try:
            shutil.copy(input, pathFake)
        except Exception as e:
            print(str(e))

# TODO commentare
# Qui i video in "Rose - Youtube Face" vengono suddivisi tra Real e Fake e viene richiamata la funzione createDataSet
# per creare i dataset per Eyeblink.
class Database:

    def __init__(self, index):


        self.data = []
        self.target = []

        if index == 0:
            # qui avviene la suddivisione dei video in Real e Fake andando a prenderli dalle sottodirectory della
            # directory root che gli passiamo
            root = 'ROSE - Youtube Face'
            for path, subdirs, files in os.walk(root):
                for name in files:
                    #scartiamo alcune tipologie di video che non ci servono
                    if not name.startswith(('Mc', 'Mu', 'Ml')):
                        # gestiamo i video genuini
                        if name.startswith('G'):
                            input = os.path.join(path, name)
                            print(input)
                            createDataSet(input, 'Real')
                        # gestiamo i video fake:
                        # 'Mf' sta per una maschera fatta di carta senza ritagli
                        # 'Ps' indica una carta stampata
                        # 'Pq' indica una carta stampata che viene fatta oscillare durante il video
                        elif name.startswith(('Mf', 'Ps', 'Pq')):
                            input = os.path.join(path, name)
                            print(input)
                            createDataSet(input, 'Fake')


if __name__ == '__main__':
    Database(0)
