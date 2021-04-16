import csv
import sys
import os
import numpy as np
import shutil
import cv2
import pandas as pd
import AntiSpoofingTrainingEvaluation


import LBP

#import dask
#import dask.array as da
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from mlxtend.plotting import plot_decision_regions
#from dask_ml.model_selection import train_test_split

csv.field_size_limit(sys.maxsize)


def get_normalized(image):
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
    return norm_image

def writeCsv( histogram , val ):
    # outstring = '\r\n'.join ([
    #     ';'.join(map( str, histogram))
    # ])
    #row_data = np.append(histogram,val)
    print(histogram)
    list =[]
    for val_array in histogram:
         list.append(val_array)
    list.append(val)


    with open('histogram.csv', 'a+') as cvsfile:
        writer = csv.writer(cvsfile, delimiter=';')
        writer.writerow(list)

        #cvsfile.write(str(val))
        #cvsfile.write('\n')
        cvsfile.close()

def convert_image_to_hist(image):

    print(image)
    image = cv2.imread(image)
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    norm_image = get_normalized(image)

    #myLBP = LBP.Local_Binary_Pattern(1, 8, norm_image)
    myLBP = LBP.Spoof_Local_Binary_Pattern(1,8,norm_image)
    new_img = myLBP.compute_lbp()
    # print(np.array(new_img).astype(np.uint8))
    # while (True):
    #     cv2.imshow('frame', np.array(new_img).astype(np.uint8))
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    hist = myLBP.createHistogram(new_img)
    return hist


    # plt.figure()
    # plt.title("Grayscale Histogram")
    # plt.xlabel("Bins")
    # plt.ylabel("# of Pixels")
    # plt.plot(hist)
    # plt.show()

def column_len_csv(filecsv):
    with open(filecsv, 'r') as f:
        reader = csv.reader(f,delimiter=';')
        for row in reader:
             return len(row)
        # first_col_len = len(next(zip(*reader)))
        # return first_col_len


def splitting_train_test(filecsv):
    num_columns = column_len_csv(filecsv)
    print(num_columns)
    #data = pd.read_csv(filecsv, header = None, usecols=[i for i in range(num_columns)])
    data = pd.read_csv(filecsv, sep=';', header= None)
    #
    # l = [i for i in range(num_columns-1)]
    # X = data[l]
    # y= data[num_columns-1]

    X,y = data.iloc[:, :-1], data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1, shuffle=True)

    return X_train, X_test, y_train, y_test

#FUNZIONE TEMPORANEA INIZIO:
def count_print_row(filecsv):


    with open(filecsv, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter=';')
        for row in reader:
            #print(" ".join(row))
            print("La lunghezza è: "+ str(len(row)))
            #print("\n\n\n")

    csvFile.close()

#FINE





    # y_probas = svm.predict_proba(X_test)
    # fpr, tpr, thresholds = roc_curve(y_train, y_probas, pos_label=0)
    # fnr = 1 - tpr
    #_____

    # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Evaluate by means of a confusion matrix
    # matrix = plot_confusion_matrix(svm, X_test, y_test,
    #                                cmap=plt.cm.Blues,
    #                                normalize='true')
    # plt.title('Confusion matrix for RBF SVM')
    # #plt.show(matrix)
    # plt.show()

    # Generate predictions
    #y_pred = svm.predict(X_test)

    # Evaluate by means of accuracy
    #accuracy = accuracy_score(y_test, y_pred)
    #print(f'Model accuracy: {accuracy}')

    # Plot decision boundary
    # feature_values = {i: 1 for i in range(2, 16384)}
    # feature_width = {i: 1 for i in range(2, 16384)}
    # #X_test = X_test.as_matrix()
    # #plot_decision_regions(X_test.values, y_test.values, clf=svm, legend=2)
    # plot_decision_regions(X_test.values, y_test.values, clf=svm,
    #                       filler_feature_values= feature_values,
    #                       filler_feature_ranges= feature_width,
    #                       res=0.02, legend=2)
    # plt.show()

def main():
    root_dir = 'Data'
    fill_csv_real = False
    fill_csv_fake = False

    try:
        os.makedirs(root_dir+'/hist_real')
    except:
        print("La directory seguente è già stata creata: "+root_dir+"/hist_real")
    try:
        os.makedirs(root_dir+'/hist_fake')
    except:
        print("La directory seguente è già stata creata: " + root_dir + "/hist_fake")

    current_real = '/Real'
    current_fake = '/Fake'

    # Qui andiamo ad inserire histogram in csv per ogni immagine Real
    if fill_csv_real == True:
        src_real = "Data" + current_real

        allFileNames = os.listdir(src_real)


        filesName = np.array(allFileNames)

        filesName = [src_real + '/' + name for name in filesName.tolist()]

        for name in filesName:
            hist_real = convert_image_to_hist(name)
            #print(len(hist_real))
            writeCsv(hist_real, 0)



        #shutil.copy(name, "Data/hist_real/")

    # Qui andiamo ad inserire histogram in csv per ogni immagine Fake
    if fill_csv_fake == True:
        src_fake = "Data" + current_fake

        allFileNames = os.listdir(src_fake)


        filesName = np.array(allFileNames)

        filesName = [src_fake + '/' + name for name in filesName.tolist()]

        for name in filesName:
             hist_real = convert_image_to_hist(name)
             writeCsv(hist_real, 1)


    #count_print_row('histogram.csv')
    X_train, X_test, y_train, y_test = splitting_train_test('histogram.csv')
    svm, y_train_score, y_test_score = AntiSpoofingTrainingEvaluation.ModelSVM(X_train, y_train, X_test, y_test).train_svm()
    print ("Y_test_pred")
    print (y_test_score)
    print ("Y_test")
    print (y_test)
    print ("X_train")
    print (X_train)
    print ("X_train_score")

    AntiSpoofingTrainingEvaluation.plot_roc_curve(y_test, y_test_score)
    FRR,SFAR=AntiSpoofingTrainingEvaluation.spoofing_scenario(y_test,y_test_score)
    print("Spoofing Scenario")
    print("FRR: ", FRR)
    print("SFAR: ", SFAR)
    FRR, FAR , HTER = AntiSpoofingTrainingEvaluation.licit_scenario(y_test,y_test_score)
    print("Licit Scenario:")
    print("FRR: ", FRR)
    print("FAR: ", FAR)
    print("HTER: ", HTER)

    return svm




###per il momento questo qui sotto non lo vedo necessario, da capire se vogliamo metterlo oppure no, a mio avviso è superfluo.
def splitting():
    # # Creating Train / Val / Test folders (One time use)
    root_dir = 'Data/'
    test_ratio = 0.20
    #posCls = '/DPN+'
    #negCls = '/DPN-'

    os.makedirs(root_dir +'/train_real')
    os.makedirs(root_dir +'/train_fake')
    os.makedirs(root_dir +'/test_fake')
    #os.makedirs(root_dir +'/val' + negCls)
    os.makedirs(root_dir +'/test_real')
    #os.makedirs(root_dir +'/test' + negCls)

    # Creating partitions of the data after shuffeling
    current_real = '/Real'
    current_fake = '/Fake'

    src_real = "Data" + current_real # Folder to copy images from

    allFileNames = os.listdir(src_real)
    np.random.shuffle(allFileNames)
    train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)* (1 - test_ratio))])


    train_FileNames = [src_real + '/' + name for name in train_FileNames.tolist()]
    #val_FileNames = [src_real+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src_real + '/' + name for name in test_FileNames.tolist()]

    print ('###REAL###')
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    #print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))
    print('###########################')

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, "Data/train_real/")

    #for name in val_FileNames:
    #    shutil.copy(name, "Data/Real/val")

    for name in test_FileNames:
        shutil.copy(name, "Data/test_real/")

    src_fake = "Data" + current_fake

    allFileNames = os.listdir(src_fake)
    np.random.shuffle(allFileNames)
    train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)* (1 - test_ratio))])

    train_FileNames = [src_fake + '/' + name for name in train_FileNames.tolist()]
    #val_FileNames = [src_real+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src_fake + '/' + name for name in test_FileNames.tolist()]


    print ('###FAKE###')
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    #print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))
    print('###########################')



    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, "Data/train_fake/")

    #for name in val_FileNames:
    #    shutil.copy(name, "Data/Real/val")

    for name in test_FileNames:
        shutil.copy(name, "Data/test_fake/")


if __name__ == '__main__':
    main()



