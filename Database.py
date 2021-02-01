import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tarfile
import cv2
import os

class Database():

    def __init__(self):
        # probe of user that are not in the gallery in percentage
        self.pn = 20

        self.data=np.load("Olivetti_faces/olivetti_faces.npy")
        self.target=np.load("Olivetti_faces/olivetti_faces_target.npy")

        # self.data = []
        # self.target = []
        # tar = tarfile.open("LFW\lfw-funneled.tgz", "r:gz")
        # counter = 0
        # for tarinfo in tar:
        #
        #     tar.extract(tarinfo.name)
        #     if tarinfo.name[-4:] == ".jpg":
        #         image = cv2.imread(tarinfo.name, cv2.IMREAD_COLOR)
        #         image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        #         self.data.append(np.array(image))
        #         counter += 1
        #
        #         name = tarinfo.name.split("/")[1]
        #         self.target.append(name)
        #     if tarinfo.isdir():
        #         pass
        #     else:
        #         os.remove(tarinfo.name)
        # tar.close()

    def split_data(self):
        unique, counts = np.unique(db.target, return_counts=True)
        occurrences = dict(zip(unique, counts))
        num_user = self.num_user()
        #numero di utenti dopo i quali si inserisce l'utente nel PG
        skipped_user = round(num_user* self.pn/100)
        print(skipped_user)
        count=0
        template=0
        gallery_data, gallery_target, probe_data,probe_target = [], [], [], []
        for i,val in enumerate(self.target):
            occ = occurrences[val]
            #print("it:",i,"count:",count,"skipped:",skipped_user,"cond:",count<skipped_user,"template:", template,"occ:",occ,"user:",val,)
            if (count<skipped_user):
                if (occ==1):
                    gallery=True
                else:
                    #numero gallery template
                    ngt = round(occ/2)
                    if (template<ngt): gallery=True
                    else: gallery=False
                template = template+1
                if (template==occ):
                    template=0
                    count = count+1
            else:
                template = template+1
                gallery=False
                if (template == occ):
                    template = 0
                    if (count==skipped_user):
                        count=0
            if (gallery):
                gallery_data.append(self.data[i])
                gallery_target.append(self.target[i])
            else:
                probe_data.append(self.data[i])
                probe_target.append(self.target[i])
        return gallery_data,gallery_target,probe_data,probe_target





    def num_user(self):
        return len(np.unique(self.target))


if __name__ == '__main__':
    db = Database()
    print("Numero utenti: ",len(np.unique(db.target)))
    print(len(db.target))

    #print("Nome utente: ",db.target[1])
    #plt.imshow(db.data[1])
    #plt.show()

    gallery_data,gallery_target,probe_data,probe_target = db.split_data()
    print("gallery:", len(gallery_data), len(gallery_target), len(np.unique(gallery_target)))
    print("probe:", len(probe_data), len(probe_target), len(np.unique(probe_target)))