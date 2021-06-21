import csv
import os
import numpy as np
import cv2
import pandas as pd
import LBP
from sklearn.model_selection import train_test_split


# converte array del db con pixel in [0,1] in array con pixel in [0,255]
def get_normalized(image):
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
    return norm_image


# conta il numero di colonne nel file csv
def column_len_csv(filecsv):
    with open(filecsv, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            return len(row)


# Qui viene effettuato lo splitting del dataset nel csv per MicroTexture

class MicroTextureSplitting:
    def __init__(self, nomeFileCsv):
        self.nomeFileCsv = nomeFileCsv

    # splitting dal dataset in train e test
    def splitting_train_test(self):
        # E' solo un test per conoscere il numero di colonne
        num_columns = column_len_csv(self.nomeFileCsv)
        print(num_columns)

        # qui prendiamo i dati da un file csv.
        data = pd.read_csv(self.nomeFileCsv, sep=';', header=None)

        # dividiamo i dati presi dal csv
        X, y = data.iloc[:, :-1], data.iloc[:, -1]

        # effettuiamo lo splitting di train e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)

        return X_train, X_test, y_train, y_test

    # vengono scritti nel csv valori dell'histogram insieme al valore reale (0) o fake (1)
    def writeCsv(self, histogram, val):
        print(histogram)
        list = []
        for val_array in histogram:
            list.append(val_array)
        list.append(val)

        with open(self.nomeFileCsv, 'a+') as cvsfile:
            writer = csv.writer(cvsfile, delimiter=';')
            writer.writerow(list)
            cvsfile.close()

    # viene presa in input un'immagine, viene calcolato il LBP e infine creato il relativo histogram
    def convert_image_to_hist(self, image):
        print(image)
        image = cv2.imread(image)

        norm_image = get_normalized(image)
        myLBP = LBP.Local_Binary_Pattern(1, 8, norm_image)
        new_img = myLBP.compute_lbp()

        hist = myLBP.createHistogram(new_img)
        return hist

    # viene effettuato lo splitting tra real e fake  e inserite le informazioni in un csv
    def splitting_real_fake(self, fill_csv_real, fill_csv_fake):

        current_real = '/Real'
        current_fake = '/Fake'

        # Qui andiamo ad inserire histogram in csv per ogni immagine Real
        if fill_csv_real:

            # In src_real, abbiamo la directory con le immagini genuine (Real)
            src_real = "MicroTextureDB" + current_real

            # Andiamo a leggere questa directory, ottenenendo una lista dei file (in questo caso immagini)
            allFileNames = os.listdir(src_real)

            filesName = np.array(allFileNames)

            # andiamo a creare una lista dove per ogni elemento, abbiamo il path compreso di file
            # per esempio /MicroTextureDB/Real/img0.jpg
            filesName = [src_real + '/' + name for name in filesName.tolist()]

            # per ogni immagine andiamo ad ottenere l'histogram relativo e lo scriviamo in csv con il valore reale (0)
            for name in filesName:
                print(name)
                hist_real = self.convert_image_to_hist(name)
                self.writeCsv(hist_real, 0)

        # Qui andiamo ad inserire histogram in csv per ogni immagine Fake
        if fill_csv_fake:
            # In src_fake, abbiamo la directory con le immagini non reali (Fake)
            src_fake = "MicroTextureDB" + current_fake
            # Andiamo a leggere questa directory, ottenenendo una lista dei file (in questo caso immagini)
            allFileNames = os.listdir(src_fake)

            filesName = np.array(allFileNames)
            # andiamo a creare una lista dove per ogni elemento, abbiamo il path compreso di file
            # per esempio /MicroTextureDB/Fake/img0.jpg
            filesName = [src_fake + '/' + name for name in filesName.tolist()]

            # per ogni immagine andiamo ad ottenere l'histogram relativo e lo scriviamo in csv con il valore reale (0)
            for name in filesName:
                print(name)
                hist_real = self.convert_image_to_hist(name)
                self.writeCsv(hist_real, 1)


def main():
    nameFileCsv = 'histogram.csv'
    MicroTextureSplitting(nameFileCsv).splitting_real_fake(False, False)


if __name__ == '__main__':
    main()
