import numpy as np
import cv2
from enum import Enum
import pandas as pd
from random import randrange
from datetime import date

import LBP
import Recognition


# Poiche' il db di Olivetti presenta degli utenti senza nome, sono stati associati ad ogni utente un'identita'
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


# percentuale di template nel probe set
probe = 50
# percentuale di utenti in PN
pn = 30


class Database:

    def __init__(self, create: bool):
        # numero di famigliari massimo per ogni utente nella gallery
        self.family_number = 2
        # se create, allora crea il db, altrimenti lo carica
        if create:
            print("CREATING NEW DB")
            # crea il db Olivetti
            data, target = self.load_db()
            # crea la lista dei cf per ogni utente
            cfs = self.defineCFlist(target)
            self.n_template_x_user = 0
            # si splitta
            self.gallery_data, self.gallery_target, self.pn_data, self.pn_target, self.pg_data, self.pg_target, \
            self.histogram_gallery_data, self.histogram_pg_data, self.histogram_pn_data = \
                self.split_gallery_probe(data, target, cfs)
            self.csv_maker(cfs)
            # Creazione dei threshold adattivi
            self.gallery_threshold = self.adaptiveThresholds()
            np.save("npy_db/gallery_data.npy", self.gallery_data)
            np.save("npy_db/gallery_target.npy", self.gallery_target)
            np.save("npy_db/gallery_thresholds.npy", self.gallery_threshold)
            np.save("npy_db/pn_data.npy", self.pn_data)
            np.save("npy_db/pn_target.npy", self.pn_target)
            np.save("npy_db/pg_data.npy", self.pg_data)
            np.save("npy_db/pg_target.npy", self.pg_target)
            np.save("npy_db/histogram_gallery_data.npy", self.histogram_gallery_data)
            np.save("npy_db/histogram_pg_data.npy", self.histogram_pg_data)
            np.save("npy_db/histogram_pn_data.npy", self.histogram_pn_data)
        else:
            print("LOADING DB")
            self.gallery_data = np.load("npy_db/gallery_data.npy")
            self.gallery_target = np.load("npy_db/gallery_target.npy")
            self.gallery_threshold = np.load("npy_db/gallery_thresholds.npy")
            self.pn_data = np.load("npy_db/pn_data.npy")
            self.pn_target = np.load("npy_db/pn_target.npy")
            self.pg_data = np.load("npy_db/pg_data.npy")
            self.pg_target = np.load("npy_db/pg_target.npy")
            self.histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
            self.histogram_pg_data = np.load("npy_db/histogram_pg_data.npy")
            self.histogram_pn_data = np.load("npy_db/histogram_pn_data.npy")
            unique, counts = np.unique(self.gallery_target, return_counts=True)
            self.n_template_x_user = counts[0]


    def adaptiveThresholds(self):
        print("ADAPTIVE THRESHOLDS")
        thresholds = []
        # lista di utenti non ripetuti
        galley_users = list(dict.fromkeys(self.gallery_target))
        # per ogni utente
        for user in galley_users:
            max_thd = -1
            # per ogni template
            for i in range(len(self.gallery_data)):
                # se i due utenti non sono uguali
                if user != self.gallery_target[i]:
                    new_thd = 0
                    # ottengo l'histogram h del template
                    hist_probe = self.histogram_gallery_data[i]
                    index = self.gallery_target.index(user)
                    # per ogni histogram dell'utente, lo comparo con h e prendo il valore di similarity piu' alto
                    for j in range(self.get_n_gallery_temp_x_user()):
                        hist_gallley = self.histogram_gallery_data[index + j]
                        diff = Recognition.compareHistogram(hist_probe, hist_gallley)
                        if diff >= new_thd:
                            new_thd = diff
                    # aggiorno il threshold piu' alto ottenuto per user e lo arrotondo
                    if new_thd > max_thd:
                        if np.round(new_thd, 2) <= new_thd:
                            max_thd = np.round(new_thd, 2) + 0.01
                        else:
                            max_thd = np.round(new_thd, 2)
            thresholds.append(max_thd)
            print("Threshold per l'utente", user, ":", max_thd)
        print("Thresholds:", thresholds)

        return thresholds

    # pn = percentuale di utenti che non sono nella gallery
    # probe = percentuale di template che sono nel probe set e non nel gallery set per lo stesso utente
    def split_gallery_probe(self, data, target, cfs):
        num_user = self.num_user(target)
        # calcola il numero di template per utente
        unique, counts = np.unique(target, return_counts=True)
        occurrences = dict(zip(unique, counts))
        # numero di utenti che non sono nella gallery
        pn_user = round(num_user * pn / 100)
        # conteggio dei template
        countTemp = 0
        # conteggio
        countUser = 0
        # gallery_target, gallery_data, pn_data, pn_target, pg_data, pg_target = [], [], [], [], [], []
        gallery_target, gallery_data, pn_data, pn_target, pg_data, pg_target, histogram_gallery_data, \
        histogram_pg_data, histogram_pn_data = [], [], [], [], [], [], [], [], []
        # prendo il numero di template dell'utente. si parte dal presupposto che tutti gli utenti
        # hanno lo stesso numero di template
        occ = occurrences[target[0]]
        # numero di template per il probe set
        self.n_template_x_user = round(occ * probe / 100)
        # per ogni utente
        for i, val in enumerate(target):
            name = cfs[countUser]
            norm_template = self.get_normalized_template(i, data)
            lbp = LBP.Local_Binary_Pattern(1, 8, norm_template)
            # se il numero del template e' minore del numero massimo di template destinati alla gallery
            # e il numero di utenti rientra negli utenti che sono nella gallery, allora inserisco il template nella
            # gallery
            if (countTemp < occ - self.n_template_x_user or occ == 1) and countUser < num_user - pn_user:
                gallery_data.append(norm_template)
                gallery_target.append(name)
                histogram_gallery_data.append(lbp.createHistogram(lbp.compute_lbp()))
            else:
                # se lo inserisco nel probe set, controllo se l'utente e' nella gallery o no
                if countUser < num_user - pn_user:
                    pg_data.append(norm_template)
                    pg_target.append(name)
                    histogram_pg_data.append(lbp.createHistogram(lbp.compute_lbp()))
                else:
                    pn_data.append(norm_template)
                    pn_target.append(name)
                    histogram_pn_data.append(lbp.createHistogram(lbp.compute_lbp()))
            countTemp += 1
            # se ho finito i template dell'utente, passo all'utente successivo
            if countTemp == occ:
                countTemp = 0
                countUser += 1
        return gallery_data, gallery_target, pn_data, pn_target, pg_data, pg_target, histogram_gallery_data, histogram_pg_data, histogram_pn_data

    # ritorna il numero di utenti nel dataset
    def num_user(self, target):
        return len(np.unique(target))

    # ritorna il numero di template nella gallery per utente
    def get_n_gallery_temp_x_user(self):
        return self.n_template_x_user

    # converte array del db con pixel in [0,1] in array con pixel in [0,255]
    def get_normalized_template(self, i, data):
        norm_image = cv2.normalize(data[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_image = norm_image.astype(np.uint8)
        return norm_image

    # carica il db
    def load_db(self):
        data = np.load("Olivetti_faces/olivetti_faces.npy")
        target = np.load("Olivetti_faces/olivetti_faces_target.npy")
        return data, target

    # realizza dataset_user.csv
    def csv_maker(self, cfs):
        dataset = []
        # numero di utenti nella gallery
        n_user = self.num_user(self.gallery_target)
        # per ogni  utente
        count = 0
        for user in cfs:
            if user in np.unique(self.gallery_target):
                row = []
                # aggiungo nome
                row.append(Olivetti_Names(count).name.replace("_", " "))
                # aggiungo il codice fiscale
                row.append(user)
                # genero la lista casuale di farmaci
                row.append(self.generateMedicineList())
                # definisco il numero di delegati
                n_family = randrange(self.family_number + 1)
                family = []
                for j in range(n_family):
                    family_member = randrange(n_user)
                    family.append(cfs[family_member])
                row.append(family)
                dataset.append(row)
                count += 1
        # trasformo in np array
        dataset = np.array(dataset)
        # e' passato almeno un mese dall'ultimo prelievo di farmaci (per motivi di testing)
        today = date.today()
        today = today.replace(month=int(today.strftime("%m")) - 1)
        df = pd.DataFrame({'User': dataset[::, 0],
                           'Codice Fiscale': dataset[::, 1],
                           'Farmaci': dataset[::, 2],
                           'Delegati': dataset[::, 3],
                           'Data': today.strftime("%d/%m/%Y")})
        # salvo in csv
        df.to_csv('dataset_user.csv')

    # realizzo dataset_medicine.csv
    def csv_medicine_maker(self):
        dataset = []
        dataset.append(["Cardioaspirina", "100 mg", 30, 1])
        dataset.append(["DAFLON", "500 mg", 30, 2])
        dataset.append(["Adalat", "60 mg", 14, 1])
        dataset.append(["Lasopranzolo", "30 mg", 28, 1])
        dataset.append(["Motilex", "0,5 mg", 30, 2])
        dataset.append(["Prefolic", "15 mg", 30, 1])
        dataset.append(["SIMESTAT", "5 mg", 20, 1])
        dataset = np.array(dataset)
        df = pd.DataFrame({'Nome': dataset[::, 0],
                           'Dosaggio': dataset[::, 1],
                           'Numero Pasticche': dataset[::, 2],
                           'Dose x giorno': dataset[::, 3],
                           })
        df.to_csv('dataset_medicine.csv')

    # generazione di un codice fiscale (fonte Wikipedia). Sono aggiunti caratteri casuali in casi particolari
    def generateCF(self, name):
        first_name = name.split(" ")[0]
        last_name = name.split(" ")[1]
        cf = ""
        total = 0
        # prime tre consonanti del cognome
        for c in enumerate(last_name):
            if c[1].lower() not in {"a", "e", "i", "o", "u", "y"}:
                cf += c[1].upper()
                total += 1
            if total == 3:
                break
        total = 0
        # prime tre consonanti
        for c in enumerate(first_name):
            if c[1].lower() not in {"a", "e", "i", "o", "u", "y"}:
                cf += c[1].upper()
                total += 1
            if total == 3:
                break
        # anno di nascita (randomico)
        for i in range(2):
            cf += str(randrange(10))
        # mese e giorno di nascita (randomico)
        month = ["A", "B", "C", "D", "E", "H", "L", "M", "P", "R", "S", "T"]
        cf += month[randrange(12)]
        day = str(randrange(31))
        # giorno di nascita (se il giorno e' composto da un solo carattere, si aggiunge prima lo zero)
        if len(day) == 1:
            day = "0" + day
        cf += day
        # stato di nascita (randomico)
        char = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M", "O", "P", "R", "S", "T", "U", "V", "Z"]
        cf += char[randrange(len(char))]
        # codice di sicurezza (randomico)
        for i in range(3):
            cf += str(randrange(10))
        cf += char[randrange(len(char))]
        # se manca una cifra, si aggiunge (ad esempio consonanti sono < 3)
        if len(cf) < 16:
            diff = 16 - len(cf)
            for i in range(diff):
                cf += char[randrange(len(char))]
        return cf

    # genera la lista dei farmaci
    def generateMedicineList(self):
        list = []
        medicines = pd.read_csv("dataset_medicine.csv")
        len = medicines.shape[0]
        # fissa un numero randomico di farmaci
        for i in range(randrange(1,len)):
            rnd = randrange(len)
            m = medicines.loc[rnd]
            medicine = m['Nome'] + " " + m['Dosaggio']
            if medicine not in list:
                list.append(medicine)
        return list

    # mostra l'immagine
    def show_image(self, img):
        while (True):
            print(img)
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # genera la lista di cf. Viene generata all'inizio perche' contiene elementi randomici
    def defineCFlist(self, list):
        cfs = []
        users = np.unique(list)
        # per ogni  utente
        for user in users:
            cfs.append(self.generateCF(Olivetti_Names(int(user)).name.replace("_", " ")))
        return cfs


if __name__ == '__main__':
    db = Database(True)
    print("gallery:", len(db.gallery_data), len(db.gallery_target), len(np.unique(db.gallery_target)))
    print("probe PG:", len(db.pg_data), len(db.pg_target), len(np.unique(db.pg_target)))
    print("probe PN:", len(db.pn_data), len(db.pn_target), len(np.unique(db.pn_target)))
    # db.csv_medicine_maker()
