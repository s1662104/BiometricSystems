import numpy as np
import tarfile
import cv2
import os
from enum import Enum
import pandas as pd
from random import randrange

class Olivetti_Names(Enum):
    Xander_Bolton = 0
    Jameson_Sierra = 1
    Myron_Kay = 2
    Liam_Cote = 3
    Ronnie_Connor = 4
    Carlton_Stubbs = 5
    Luis_Richards = 6
    Carolina_Aldred = 7
    Esme_Suarez = 8
    Giulia_Sutherland = 9
    Eduardo_Villalobos = 10
    Maxime_Koch = 11
    Francis_Oconnor = 12
    Omari_Bellamy = 13
    Harlan_Chapman = 14
    Kirk_Meadows = 15
    Darren_Haynes = 16
    Jim_Bostock = 17
    Vladimir_Sumner = 18
    Keyan_Suarez = 19
    Curtis_Willis = 20
    Jae_Watt = 21
    Merlin_Heaton = 22
    Matthew_Lloyd = 23
    Finn_Curry = 24
    Toby_Cabrera = 25
    Martin_Dickinson = 26
    Cobie_Whitfield = 27
    Elyas_Vo = 28
    Dominick_Deacon = 29
    Tobey_Davis = 30
    Caitlyn_Schneider = 31
    Rayan_Diaz = 32
    Phillip_Donaldson = 33
    Karina_Anthony = 34
    Stan_Curran = 35
    Aston_Andrews = 36
    Junior_Pham = 37
    Konnor_Buck = 38
    Martyn_Mays = 39

class Medicine(Enum):
    Cardioaspirina = 0
    DAFLON = 1
    Cardiol = 2
    CardioPlus = 3
    Adalat = 4
    Lasopranzolo = 5
    Motilex = 6
    Chetogerd = 7
    Tachipirina = 8

class Database():

    def __init__(self, db_index=None):
        self.db_index = db_index
        # numero di famigliari massimo per ogni utente
        self.family_number = 2
        if self.db_index is not None:
            data, target = self.load_db()
            self.gallery_data, self.gallery_target, self.pn_data, self.pn_target, self.pg_data, self.pg_target = \
                self.split_gallery_probe(data, target )
            np.save("npy_db/gallery_data.npy",self.gallery_data)
            np.save("npy_db/gallery_target.npy",self.gallery_target)
            np.save("npy_db/pn_data.npy", self.pn_data)
            np.save("npy_db/pn_target.npy", self.pn_target)
            np.save("npy_db/pg_data.npy", self.pg_data)
            np.save("npy_db/pg_target.npy", self.pg_target)
        else:
            self.gallery_data = np.load("npy_db/gallery_data.npy")
            self.gallery_target = np.load("npy_db/gallery_target.npy")
            self.pn_data = np.load("npy_db/pn_data.npy")
            self.pn_target = np.load("npy_db/pn_target.npy")
            self.pg_data = np.load("npy_db/pg_data.npy")
            self.pg_target = np.load("npy_db/pg_target.npy")

    # pn = percentuale di utenti che non sono nella gallery
    # probe = percentuale di template che sono nel probe set e non nel gallery set per lo stesso utente
    def split_gallery_probe(self, data,target, pn=30, probe=50):
        num_user = self.num_user(target)
        unique, counts = np.unique(target, return_counts=True)
        occurrences = dict(zip(unique, counts))
        pn_user = round(num_user * pn / 100)
        count = 0
        countUser = 0
        gallery_target, gallery_data, pn_data, pn_target, pg_data, pg_target = [], [], [], [], [], []
        for i, val in enumerate(target):
            occ = occurrences[val]
            n_probe_temp = round(occ * probe / 100)
            if self.db_index == 0:
                name = Olivetti_Names(int(target[i])).name.replace("_"," ")
            else:
                name = target[i]
            if (count < occ - n_probe_temp or occ == 1) and countUser < num_user - pn_user:
                gallery_data.append(self.get_normalized_template(i, data))
                gallery_target.append(name)
            else:
                if countUser < num_user - pn_user:
                    pg_data.append(self.get_normalized_template(i, data))
                    pg_target.append(name)
                else:
                    pn_data.append(self.get_normalized_template(i, data))
                    pn_target.append(name)
            count += 1
            if count == occ:
                count = 0
                countUser += 1
        return gallery_data, gallery_target, pn_data, pn_target, pg_data, pg_target

    def num_user(self, target):
        return len(np.unique(target))

    # converte array del db con pixel in [0,1] in array con pixel in [0,255]
    def get_normalized_template(self, i, data):
        norm_image = cv2.normalize(data[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_image = norm_image.astype(np.uint8)
        if self.db_index == 1:
            norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
        return norm_image

    def load_db(self):
        data = []
        target = []
        if self.db_index == 0:
            data = np.load("Olivetti_faces/olivetti_faces.npy")
            target = np.load("Olivetti_faces/olivetti_faces_target.npy")
        elif self.db_index == 1:
            tar = tarfile.open("LFW/lfw-funneled.tgz", "r:gz")
            counter = 0
            for tarinfo in tar:
                tar.extract(tarinfo.name)
                if tarinfo.name[-4:] == ".jpg":
                    image = cv2.imread(tarinfo.name, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
                    data.append(np.array(image))
                    counter += 1
                    name = tarinfo.name.split("/")[1]
                    target.append(name)
                if tarinfo.isdir():
                    pass
                else:
                    os.remove(tarinfo.name)
            tar.close()
        else:
            print("VALORE NON VALIDO!")
        return data, target

    def csv_maker(self):
        dataset = []
        n_user = self.num_user(self.gallery_target)
        users = np.unique(self.gallery_target)
        for user in users:
            row = []
            row.append(user)
            row.append(self.generateCF(user))
            row.append(self.generateMedicineList())
            n_family = randrange(self.family_number+1)
            family = []
            for j in range(n_family):
                family_member = randrange(n_user)
                family.append(users[family_member])
            row.append(family)
            dataset.append(row)
        dataset = np.array(dataset)
        df = pd.DataFrame({'User': dataset[::, 0],
                           'Codice Fiscale': dataset[::, 1],
                           'Farmaci': dataset[::, 2],
                           'Delegati': dataset[::, 3],
                           })
        df.to_csv('dataset_farmaci.csv')

    # generazione di un codice fiscale (fonte Wikipedia). Sono aggiunti caratteri casuali in casi particolari
    def generateCF(self,name):
        first_name = name.split(" ")[0]
        last_name = name.split(" ")[1]
        cf = ""
        total = 0
        for c in enumerate(last_name):
            if c[1].lower() not in {"a", "e", "i", "o", "u", "y"}:
                cf += c[1].upper()
                total += 1
            if total == 3:
                break
        total = 0
        for c in enumerate(first_name):
            if c[1].lower() not in {"a", "e", "i", "o", "u", "y"}:
                cf += c[1].upper()
                total += 1
            if total == 3:
                break
        for i in range(2):
            cf += str(randrange(10))
        month = ["A", "B", "C", "D", "E", "H", "L", "M", "P", "R", "S", "T"]
        cf += month[randrange(12)]
        day = str(randrange(31))
        if len(day)==1:
            day = "0"+day
        cf += day
        char = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M", "O", "P", "R", "S", "T", "U", "V", "Z"]
        cf += char[randrange(len(char))]
        for i in range(3):
            cf += str(randrange(10))
        cf += char[randrange(len(char))]
        if len(cf) < 16:
            diff = 16 - len(cf)
            for i in range(diff):
                cf += char[randrange(len(char))]
        return cf

    def generateMedicineList(self):
        list = []
        for i in range(randrange(len(Medicine))):
            medicine = Medicine(randrange(len(Medicine))).name
            if medicine not in list:
                list.append(medicine)
        return list

    def show_image(self, img):
        while(True):
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    db = Database()
    print("gallery:", len(db.gallery_data), len(db.gallery_target), len(np.unique(db.gallery_target)))
    print("probe PG:", len(db.pg_data), len(db.pg_target), len(np.unique(db.pg_target)))
    print("probe PN:", len(db.pn_data), len(db.pn_target), len(np.unique(db.pn_target)))

    db.csv_maker()

