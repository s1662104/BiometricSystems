import numpy as np
import tarfile
import cv2
import os


class Database():

    def __init__(self, db_index):

        # probe of user that are not in the gallery in percentage
        self.db_index = db_index
        self.data = []
        self.target = []

        if self.db_index == 0:
            self.olivettiDB()
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

    def olivettiDB(self):
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
            if (template < div or occ == 1) and countTest < num_user - test_no_train:
                train_data.append(self.get_normalized_template(i))
                train_target.append(self.target[i])
            else:
                test_data.append(self.get_normalized_template(i))
                test_target.append(self.target[i])
                # se tale condizione e' vera, significa che in test ci vanno tutti i template dell'i-esimo utente
                if countTest >= num_user - test_no_train:
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

    # pn = percentuale di utenti che non sono nella gallery
    # probe = percentuale di template che sono nel probe set e non nel gallery set per lo stesso utente
    def split_gallery_probe(self, pn=30, probe=50):
        num_user = self.num_user()
        unique, counts = np.unique(self.target, return_counts=True)
        occurrences = dict(zip(unique, counts))
        pn_user = round(num_user * pn / 100)
        count = 0
        countUser = 0
        gallery_target, gallery_data, pn_data, pn_target, pg_data, pg_target = [], [], [], [], [], []
        for i, val in enumerate(self.target):
            occ = occurrences[val]
            n_probe_temp = round(occ * probe / 100)
            if (count < occ - n_probe_temp or occ == 1) and countUser < num_user - pn_user:
                gallery_data.append(self.get_normalized_template(i))
                gallery_target.append(self.target[i])
            else:
                if countUser < num_user - pn_user:
                    pg_data.append(self.get_normalized_template(i))
                    pg_target.append(self.target[i])
                else:
                    pn_data.append(self.get_normalized_template(i))
                    pn_target.append(self.target[i])
            count += 1
            if count == occ:
                count = 0
                countUser += 1
        return gallery_data, gallery_target, pn_data, pn_target, pg_data, pg_target

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

    def load_db(self):
        pass

if __name__ == '__main__':
    db = Database(0)
    print("Numero utenti: ",len(np.unique(db.target)))
    print("Template:", len(db.target))

    #COME SALVARE E RICARICARE IL SET
    #np.save("X_train.npy",X_train)
    #np.save("Y_train.npy",X_test)

    #X_train = np.load("X_train.npy")
    #Y_train = np.load("Y_train.npy")

    gallery_data, gallery_target, pn_data, pn_target, pg_data, pg_target = db.split_gallery_probe()
    print("gallery:", len(gallery_data), len(gallery_target), len(np.unique(gallery_target)))
    print("probe PG:", len(pg_data), len(pg_target), len(np.unique(pg_target)))
    print("probe PN:", len(pn_data), len(pn_target), len(np.unique(pn_target)))

