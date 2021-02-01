import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tarfile
import cv2
import os

class Database():
    def __init__(self):
        # self.data=np.load("Olivetti_faces/olivetti_faces.npy")
        # self.target=np.load("Olivetti_faces/olivetti_faces_target.npy")

        self.data = []
        self.target = []
        tar = tarfile.open("LFW\lfw-funneled.tgz", "r:gz")
        counter = 0
        for tarinfo in tar:

            tar.extract(tarinfo.name)
            if tarinfo.name[-4:] == ".jpg":
                image = cv2.imread(tarinfo.name, cv2.IMREAD_COLOR)
                image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
                self.data.append(np.array(image))
                counter += 1

                name = tarinfo.name.split("/")[1]
                self.target.append(name)
            if tarinfo.isdir():
                pass
            else:
                os.remove(tarinfo.name)
        tar.close()








if __name__ == '__main__':
    db = Database()
    print("Numero utenti: ",len(np.unique(db.target)))

    print("Nome utente: ",db.target[1])

    plt.imshow(db.data[1])
    plt.show()
