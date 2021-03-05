import numpy as np
import tarfile
import cv2
import os

class Database():

    def __init__(self, db_index=None):
        self.db_index = db_index
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
            if (count < occ - n_probe_temp or occ == 1) and countUser < num_user - pn_user:
                gallery_data.append(self.get_normalized_template(i, data))
                gallery_target.append(target[i])
            else:
                if countUser < num_user - pn_user:
                    pg_data.append(self.get_normalized_template(i, data))
                    pg_target.append(target[i])
                else:
                    pn_data.append(self.get_normalized_template(i, data))
                    pn_target.append(target[i])
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

if __name__ == '__main__':
    db = Database()
    print("gallery:", len(db.gallery_data), len(db.gallery_target), len(np.unique(db.gallery_target)))
    print("probe PG:", len(db.pg_data), len(db.pg_target), len(np.unique(db.pg_target)))
    print("probe PN:", len(db.pn_data), len(db.pn_target), len(np.unique(db.pn_target)))

