import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tarfile
import cv2
import os
import csv
import shutil
from sklearn import metrics
from sklearn.svm import SVC
import LBP
import Main


def createDataSet(input, val, name, replayAttack, eyeBlink):

    if replayAttack == True:
        cap = cv2.VideoCapture(input)
        pathReal = 'Data/ReplayAttack/Real/'
        pathFake = 'Data/ReplayAttack/Fake/'
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
                    try:
                        cv2.imwrite(pathReal+ name +"_{}.jpg".format(counter), crop)
                    except Exception as e:
                        print(str(e))
                elif val == 'Fake':
                    if not os.path.exists(pathFake):
                        os.makedirs(pathFake)
                    try:
                        cv2.imwrite(pathFake+name+"_{}.jpg".format(counter), crop)
                    except Exception as e:
                        print(str(e))
            break
        cap.release()
        cv2.destroyAllWindows()
    if eyeBlink == True:
        pathReal = 'Data/EyeBlink/Real/'
        pathFake = 'Data/EyeBlink/Fake/'
        if val == 'Real':
                if not os.path.exists(pathReal):
                    os.makedirs(pathReal)
                try:
                    shutil.copy(input,pathReal)
                except Exception as e:
                        print(str(e))
        elif val == 'Fake':
                if not os.path.exists(pathFake):
                    os.makedirs(pathFake)
                try:
                    shutil.copy(input, pathFake)
                except Exception as e:
                    print(str(e))
        #filename = 'img_' + str(counter) +'.jpg'
        #with open(os.path.join(path, filename), 'wb') as temp_file:

            #temp_file.write(crop)




#def createDataSetEyeBlink(input, val, subdir):




class Database():

    def __init__(self, index):
        #index = 0 Database ROSE
        #index = 1 Database Eyeblink8
        # probe of user that are not in the gallery in percentage
        #self.pn = 20    #numero utenti dopo il quale inserisce un utente solo nel probe set
        #self.db_index = db_index
        self.data = []
        self.target = []
        if index == 0:

            root = 'ROSE - Youtube Face'
            input = None


            counter = 0
            for path, subdirs, files in os.walk(root):
                for name in files:
                    if not name.startswith(('Mc','Mf','Mu','Ml')):
                        if name.startswith('G'):
                            input = os.path.join(path, name)
                            print(input)
                            createDataSet(input, 'Real', name, True, False)
                        else:
                            input = os.path.join(path, name)
                            print(input)
                            createDataSet(input, 'Fake', name, True, False)
        elif index == 1:
            input = None
            root = 'ROSE - Youtube Face'
            for path, subdirs, files in os.walk(root):
                for name in files:
                    if not name.startswith(('Mc','Mu','Ml')):
                        if name.startswith('G'):
                            input = os.path.join(path,name)
                            print(input)
                            createDataSet(input, 'Real', name, False , True)
                        elif name.startswith(('Mf','Ps','Pq')) :
                            input = os.path.join(path,name)
                            print(input)
                            createDataSet(input, 'Fake', name, False, True)












if __name__ == '__main__':

    #Database(0)
    Database(1)


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