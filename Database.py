import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tarfile
import cv2
import os
import csv

from sklearn.svm import SVC

import LBP

class Database():

    def __init__(self, db_index):

        # probe of user that are not in the gallery in percentage
        self.pn = 20    #numero utenti dopo il quale inserisce un utente solo nel probe set
        self.db_index = db_index
        self.data = []
        self.target = []

        if self.db_index == 0:
            self.secondDB()
        elif self.db_index == 1:
            tar = tarfile.open("LFW/lfw-funneled.tgz", "r:gz")
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
        else:
            print("VALORE NON VALIDO!")

    def secondDB(self):
        imgs = np.load("Olivetti_faces/olivetti_faces.npy")
        imgs.shape
        type(imgs)
        self.data = imgs

        targets = np.load("Olivetti_faces/olivetti_faces_target.npy")
        targets.shape
        type(targets)
        self.target = targets

    # 0.7 = 30% degli utenti e' nel test ma non nel train
    def split_data(self,percTest=30, percPN=15):
        train_data, train_target, test_data,test_target,gallery_data, gallery_target, probe_data,probe_target = [], [], [], [], [], [], [], []
        num_user = self.num_user()
        test_no_train = round(num_user * percTest / 100)
        probe_no_gallery = round(num_user * percPN / 100)
        print("Numero utenti in test ma non in train:", test_no_train, "Numero utenti in probe set ma non in gallery", probe_no_gallery)
        countTest = 0
        countPN = 0
        template = 0
        unique, counts = np.unique(self.target, return_counts=True)
        occurrences = dict(zip(unique, counts))
        for i, val in enumerate(self.target):
            occ = occurrences[val]
            div = round(occ/2)
            if (template<div or occ==1) and countTest<num_user - test_no_train:
                train_data.append(self.get_normalized_template(i))
                train_target.append(self.target[i])
            else:
                test_data.append(self.get_normalized_template(i))
                test_target.append(self.target[i])
                # se tale condizione e' vera, significa che in test ci vanno tutti i template dell'i-esimo utente
                if countTest>=num_user - test_no_train:
                    divT = div
                else:
                    divT = round(div/2)
                if ((countTest < num_user - test_no_train and template-div < divT) or
                        (countTest >= num_user - test_no_train and template < div)) or occ==1:
                    gallery_data.append(self.get_normalized_template(i))
                    gallery_target.append(self.target[i])
                else:
                    probe_data.append(self.get_normalized_template(i))
                    probe_target.append(self.target[i])
            template += 1
            if template == occ:
                template = 0
                countTest += 1

        return train_data, train_target, test_data, test_target, gallery_data, gallery_target, probe_data, probe_target

    def num_user(self):
        return len(np.unique(self.target))

    def get_template(self,i):
        return self.data[i]

    # converte array del db con pixel in [0,1] in array con pixel in [0,255]
    def get_normalized_template(self, i):
        norm_image = cv2.normalize(self.get_template(i), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_image = norm_image.astype(np.uint8)
        if self.db_index == 1:
            norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
        return norm_image

    def get_target(self,i):
        return self.target[i]

    def createCSV(self):
        train_data, train_target, test_data, test_target, gallery_data, gallery_target, probe_data, probe_target = self.split_data()

        data_list = [[]]*(len(train_data)+1)
        data_list[0] = ['Image', 'Target']
        for i in range(1,len(data_list)):
            data_list[i] = [train_data[i-1].tolist(), train_target[i-1]]
        with open('train.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(data_list)

        data_list = [[]] * (len(test_data) + 1)
        data_list[0] = ['Image', 'Target']
        for i in range(1,len(data_list)):
            data_list[i] = [test_data[i-1].tolist(), test_target[i-1]]
        with open('test.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(data_list)

        data_list = [[]] * (len(gallery_data) + 1)
        data_list[0] = ['Image', 'Target']
        for i in range(1, len(data_list)):
            data_list[i] = [gallery_data[i - 1].tolist(), gallery_target[i - 1]]
        with open('gallery.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(data_list)

        data_list = [[]] * (len(probe_data) + 1)
        data_list[0] = ['Image', 'Target']
        for i in range(1, len(data_list)):
            data_list[i] = [probe_data[i - 1].tolist(), probe_target[i - 1]]
        with open('probe.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(data_list)

        return

if __name__ == '__main__':
    db = Database(0)
    print("Numero utenti: ",len(np.unique(db.target)))
    print("Template:", len(db.target))
    classifier = SVC(kernel='rbf', random_state=1)
    train_data, train_target, test_data, test_target, gallery_data, gallery_target, probe_data, probe_target = db.split_data()

    #np.save("X_train.npy",train_data)
    #np.save("Y_train.npy",train_target)

    #X_train = np.load("X_train.npy")
    #Y_train = np.load("Y_train.npy")
    X_train = [0]*len(train_data)

    for i in range(0, len(train_data)):
        lbp = LBP.Local_Binary_Pattern(1, 8, train_data[i])
        new_img = lbp.compute_lbp()
        X_train[i] = lbp.createHistogram(new_img)

    classifier.fit(X_train, train_target)

    #X_train = train['Image'].array.to_numpy().tolist()
    #Y_train = train['Target'].array.to_numpy().tolist()

    # print(db.get_template(1))
    #
    # data = db.get_normalized_template(1)
    # print(data)
    # while(True):
    #     cv2.imshow('frame', data)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # train_data, train_target, test_data,test_target,gallery_data, gallery_target, probe_data, probe_target = \
    #     db.split_data()
    # print("train:", len(train_data), len(train_target), len(np.unique(train_target)))
    # print("test:", len(test_data), len(test_target), len(np.unique(test_target)))
    # print("gallery:", len(gallery_data), len(gallery_target), len(np.unique(gallery_target)))
    # print("probe:", len(probe_data), len(probe_target), len(np.unique(probe_target)))