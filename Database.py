import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tarfile
import cv2
import os
import csv

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.model_selection import train_test_split
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
    def split_data(self,percTest=30):
        train_data, train_target, test_data,test_target,gallery_data, gallery_target, pg_data, pg_target, pn_data,\
            pn_target = [], [], [], [], [], [], [], [], [], []
        num_user = self.num_user()
        test_no_train = round(num_user * percTest / 100)
        print("Numero utenti in test ma non in train:", test_no_train)
        countTest = 0
        template = 0
        unique, counts = np.unique(self.target, return_counts=True)
        occurrences = dict(zip(unique, counts))
        for i, val in enumerate(self.target):
            occ = occurrences[val]
            div = round(occ/2)
            if (template<div or occ==1) and countTest < num_user - test_no_train:
                train_data.append(self.get_normalized_template(i))
                train_target.append(self.target[i])
            else:
                test_data.append(self.get_normalized_template(i))
                test_target.append(self.target[i])
                #se tale condizione e' vera, significa che in test ci vanno tutti i template dell'i-esimo utente
                if countTest>=num_user - test_no_train:
                    divT = div
                else:
                    divT = round(div/2)
                if (countTest < num_user - test_no_train and template-div < divT) or occ==1:
                    gallery_data.append(self.get_normalized_template(i))
                    gallery_target.append(self.target[i])
                elif countTest < num_user - test_no_train:
                    pg_data.append(self.get_normalized_template(i))
                    pg_target.append(self.target[i])
                else:
                    pn_data.append(self.get_normalized_template(i))
                    pn_target.append(self.target[i])
            template += 1
            if template == occ:
                template = 0
                countTest += 1

        return train_data, train_target, test_data, test_target, gallery_data, gallery_target, pg_data, pg_target, pn_data, pn_target

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

if __name__ == '__main__':
    db = Database(0)
    print("Numero utenti: ",len(np.unique(db.target)))
    print("Template:", len(db.target))

    X = [0]*len(db.data)
    for i in range(0,len(db.data)):
        lbp = LBP.Local_Binary_Pattern(1, 8, db.get_normalized_template(i))
        new_img = lbp.compute_lbp()
        X[i] = lbp.createHistogram(new_img)
    X_train, X_test, y_train, y_test = train_test_split(X, db.target, test_size=0.25, random_state=42)
    classifier = SVC(kernel='rbf', random_state=1)
    classifier.fit(X_train, y_train)        #Train the model using the training sets
    y_pred = classifier.predict(X_test)       #Predict the response for test dataset
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))        #Model Accuracy: how often is the classifier correct?

    # train_data, train_target, test_data, test_target, gallery_data, gallery_target, pg_data, pg_target, pn_data, pn_target = db.split_data()
    # X_train = [0] * len(train_data)
    # for i in range(0, len(train_data)):
    #     lbp = LBP.Local_Binary_Pattern(1, 8, train_data[i])
    #     new_img = lbp.compute_lbp()
    #     X_train[i] = lbp.createHistogram(new_img)
    #
    # X_test = [0] * len(test_data)
    # for i in range(0, len(test_data)):
    #    lbp = LBP.Local_Binary_Pattern(1, 8, test_data[i])
    #    new_img = lbp.compute_lbp()
    #    X_test[i] = lbp.createHistogram(new_img)
    #
    # pca = RandomizedPCA(n_components=50, whiten=True).fit(X_train)
    # X_train_pca = pca.transform(X_train)
    # X_test_pca = pca.transform(X_test)
    # classifier = SVC(kernel='rbf', random_state=1)
    # classifier.fit(X_train_pca, train_target)
    # Predict the response for test dataset
    # y_pred = classifier.predict(X_test_pca)
    # Model Accuracy: how often is the classifier correct?
    #print("Accuracy:", metrics.accuracy_score(test_target, y_pred))

    #COME SALVARE E RICARICARE IL SET
    #np.save("X_train.npy",X_train)
    #np.save("Y_train.npy",X_test)

    #X_train = np.load("X_train.npy")
    #Y_train = np.load("Y_train.npy")

    # print("train:", len(train_data), len(train_target), len(np.unique(train_target)))
    # print("test:", len(test_data), len(test_target), len(np.unique(test_target)))
    # print("gallery:", len(gallery_data), len(gallery_target), len(np.unique(gallery_target)))
    # print("probe PG:", len(pg_data), len(pg_target), len(np.unique(pg_target)))
    # print("probe PN:", len(pn_data), len(pn_target), len(np.unique(pn_target)))

