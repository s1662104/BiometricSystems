import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tarfile
import cv2
import os
import csv

from sklearn import metrics
from sklearn.svm import SVC
import LBP
import Main


def createDataSet(input, val, subdir):
    cap = cv2.VideoCapture(input)
    pathReal = 'Data/Real/'
    pathFake = 'Data/Fake/'
    counter = 0
    crop = None
    vis = None

    while ( True ):

        ret, frame = cap.read()

        vis = frame.copy()

        if vis is None:
         break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = Main.detect_face(gray,vis)

        if crop is not None:
            counter += 1
            if val == 'Real':
                if not os.path.exists(pathReal):
                    os.makedirs(pathReal)
                cv2.imwrite(pathReal+"crop{}.jpg".format(counter), crop)
            elif val == 'Fake':
                if not os.path.exists(pathFake):
                    os.makedirs(pathFake)
                cv2.imwrite(pathFake+"crop{}.jpg".format(subdir,counter), crop)
        #filename = 'img_' + str(counter) +'.jpg'
        #with open(os.path.join(path, filename), 'wb') as temp_file:

            #temp_file.write(crop)


    cap.release()
    cv2.destroyAllWindows()

class Database():

    def __init__(self):

        # probe of user that are not in the gallery in percentage
        #self.pn = 20    #numero utenti dopo il quale inserisce un utente solo nel probe set
        #self.db_index = db_index
        self.data = []
        self.target = []
        root = 'ROSE - Youtube Face'
        input = None

        #if self.db_index == 0:
        #    self.secondDB()
        #elif self.db_index == 1:
            #tar = tarfile.open("LFW/lfw-funneled.tgz", "r:gz")
        counter = 0
        for path, subdirs, files in os.walk(root):
            for name in files:
                if not name.startswith(('Mc','Mf','Mu','Ml')):
                    if name.startswith('G'):
                        input = os.path.join(path, name)
                        print(path)
                        createDataSet(input,'Real', path)
                    else:
                        input = os.path.join(path, name)
                        createDataSet(input,'Fake', path)

        #         tar.extract(tarinfo.name)
        #         if tarinfo.name[-4:] == ".jpg":
        #             image = cv2.imread(tarinfo.name, cv2.IMREAD_COLOR)
        #             image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        #             self.data.append(np.array(image))
        #             counter += 1
        #             name = tarinfo.name.split("/")[1]
        #             self.target.append(name)
        #         if tarinfo.isdir():
        #             pass
        #         else:
        #             os.remove(tarinfo.name)
        #     tar.close()
        # else:
        #     print("VALORE NON VALIDO!")





if __name__ == '__main__':

    Database()


    #db = Database(0)
    #print("Numero utenti: ",len(np.unique(db.target)))
    #print("Template:", len(db.target))
    #classifier = SVC(kernel='rbf', random_state=1)
    #train_data, train_target, test_data, test_target, gallery_data, gallery_target, pg_data, pg_target, pn_data, pn_target = db.split_data()

    # X_train = [0] * len(train_data)
    # for i in range(0, len(train_data)):
    #     lbp = LBP.Local_Binary_Pattern(1, 8, train_data[i])
    #     new_img = lbp.compute_lbp()
    #     X_train[i] = lbp.createHistogram(new_img)
    #
    # X_test = [0] * len(test_data)
    # for i in range(0, len(test_data)):
    #     lbp = LBP.Local_Binary_Pattern(1, 8, test_data[i])
    #     new_img = lbp.compute_lbp()
    #     X_test[i] = lbp.createHistogram(new_img)

    #COME SALVARE E RICARICARE IL SET
    #np.save("X_train.npy",X_train)
    #np.save("Y_train.npy",X_test)

    #X_train = np.load("X_train.npy")
    #Y_train = np.load("Y_train.npy")

    # #Train the model using the training sets
    # classifier.fit(X_train, train_target)
    #
    # #Predict the response for test dataset
    # y_pred = classifier.predict(X_test)
    #
    # #Model Accuracy: how often is the classifier correct?
    # print("Accuracy:", metrics.accuracy_score(test_target, y_pred))