import numpy as np
import cv2
import Database


class Local_Binary_Pattern:

    def __init__(self, radius, neighborhood, img):
        self.radius = radius
        self.neighborhood = neighborhood
        # il raggio non puo' essere minore di 1, il vicinato non puo' essere minore di 4 e deve essere una potenza di 2
        # questo semplifica l'ordinamento dei punti, sapendo qual è in genere l'angolo in alto a sx
        if self.radius <= 0 or self.neighborhood < 4 or (self.neighborhood & (self.neighborhood - 1)) != 0:
            raise Exception("Input error")
        self.img = img

    # calcola l'immagine lbp
    def compute_lbp(self):
        # crea la nuova immagine
        new_img = [[0 for y in range(self.img.shape[0])] for x in range(self.img.shape[1])]
        # per ogni pixel
        for i in range(0, self.img.shape[0]):
            for j in range(0, self.img.shape[1]):
                # trova i pixel del vicinato
                pixels = self.find_neighbors(i, j)
                # definisce il pattern
                pattern = np.where(pixels > self.img[i][j], 1, 0)
                value = 0
                count = 0
                # somma pesata con le potenze di due
                for k in pattern:
                    value += k * 2 ** count
                    count += 1
                new_img[i][j] = value % 256
        return new_img

    def find_neighbors(self, cx, cy):
        # definire il cerchio composto da self.neighborhood punti
        angles_array = 2 * np.pi / self.neighborhood
        alpha = np.arange(0, 2 * np.pi, angles_array)
        # ordiniamo  in modo tale da partire dall'angolo in alto a sx e procedere verso dx
        alpha = self.sort_points(alpha)
        # calcolare coppia di seno e coseno per ogni punto
        s_points = np.array([-np.sin(alpha), np.cos(alpha)]).transpose()
        s_points *= self.radius
        # per ogni punto
        pixels = []
        for x, y in s_points:
            # analizziamo la parte frazionaria, considerando solo i primi quattro valori.
            # Se questi sono 0, allora si riferiscono ai punti nei gradi
            # 0,90,180,270, che sono gli unici a trovarsi al centro del pixel.
            # In questo caso, non hanno bisogno di interpolazione
            x = np.round(x, 4)
            y = np.round(y, 4)
            x_fract = x - np.round(x)
            y_fract = y - np.round(y)
            if x_fract == 0 and y_fract == 0:
                coorx = int(x)
                coory = int(y)
                if self.check_border(cx + coorx, cy + coory):
                    pixels.append(0)
                else:
                    pixels.append(self.img[cx + coorx][cy + coory])
            # i punti seguenti non cadono al centro di un pixel
            else:
                # si prende la parte frazionaria e si arrotonda per eccesso e per difetto
                x_c = np.ceil(x).astype(int)
                y_c = np.ceil(y).astype(int)
                x_f = np.floor(x).astype(int)
                y_f = np.floor(y).astype(int)
                # ordino x1 e x2 in modo tale che x1 < x2 e lo stesso vale per y1 e y2
                if x_c == 0:
                    x1 = x_c
                    x2 = x_f
                else:
                    x1 = x_f
                    x2 = x_c
                if y_c == 0:
                    y1 = y_c
                    y2 = y_f
                else:
                    y1 = y_f
                    y2 = y_c
                # controllo i bordi: se il punto esce fuori dal bordo gli associo il valore 0
                if self.check_border(cx + x1, cy + y1) or self.check_border(cx + x2, cy + y2):
                    pixels.append(0)
                else:
                    value = self.bilinear_interpolation(x1, y1, x2, y2, cx, cy, x, y)
                    # si arrotonda il valore
                    pixels.append(np.round(value).astype(int))
        return pixels

    # x1,y1 rappresentano le coordinate più piccole
    # x2, y2 rappresentano le coordinate più grandi
    # cx, cy rappresentano le coordinate del pixel centrale nell'immagine
    # x, y rappresentano le coordinate del punto il cui valore è da definire
    def bilinear_interpolation(self, x1, y1, x2, y2, cx, cy, x, y):
        Q11 = self.img[cx + x1][cy + y1]
        Q21 = self.img[cx + x1][cy + y2]
        Q12 = self.img[cx + x2][cy + y1]
        Q22 = self.img[cx + x2][cy + y2]
        dem = (x2 - x1) * (y2 - y1)
        return (Q11 * (x2 - x) * (y2 - y) / dem) + (Q21 * (x - x1) * (y2 - y) / dem) + \
               (Q12 * (x2 - x) * (y - y1) / dem) + (Q22 * (x - x1) * (y - y1) / dem)

    # si ordinano i punti partendo dall'angolo in altro a sx e procedendo verso dx
    # [0. 0.78539816 1.57079633 2.35619449 3.14159265 3.92699082 4.71238898 5.49778714]
    # 2.35619449 rappresenta l'angolo di 135 gradi. S
    def sort_points(self, alpha):
        # l'angolo in alto a sx sarà 135. se il vicinato > 4, altrimenti si parte da 90 gradi
        if self.neighborhood > 4:
            left_point = np.where(np.degrees(alpha) == 135.)[0][0]
        else:
            left_point = np.where(np.degrees(alpha) == 90.)[0][0]
        new_alpha = []
        # parto dall'angolo in alto a sx e vado verso dx (quindi nella lista vado verso lo 0)
        # dopo parto dalla fine della lista e vado verso il centro della lista
        # questa tecnica nasce da una dall'analisi dei risultati ottenuti
        count = left_point
        while len(new_alpha) != self.neighborhood:
            new_alpha.append(alpha[count])
            if np.degrees(alpha[count]) == 0.:
                count = self.neighborhood
            count -= 1
        return new_alpha

    # controlla se fuoriesce dal bordo dell'immagine
    def check_border(self, x, y):
        return x < 0 or x >= self.img.shape[0] or y < 0 or y >= self.img.shape[1]

    # crea l'histogram
    def createHistogram(self, new_img, grid_x=8, grid_y=8):
        histogram = []

        # controlla le dimensioni della immagine LBP ottenuta
        if len(new_img) == 0:
            raise Exception("Input error")

        # Ottiene le dimensioni dell'immagine
        h = len(new_img)
        w = len(new_img[0])

        # ottiene le dimensioni di ogni regione
        gridWidth = int(w / grid_x)
        gridHeight = int(h / grid_y)

        # calcola l'histogram per ogni regione
        for gx in range(0, grid_x):
            for gy in range(0, grid_y):

                # prepara l'histogram vuoto
                regionHistogram = [0] * 256

                # definisce le posizioni iniziali e finale per il seguente loop
                startPosX = gx * gridWidth
                startPosY = gy * gridHeight
                endPosX = (gx + 1) * gridWidth
                endPosY = (gy + 1) * gridHeight

                # crea l'histogram per la regione corrente
                for x in range(startPosX, endPosX):
                    for y in range(startPosY, endPosY):
                        if new_img[x][y] < len(regionHistogram):
                            regionHistogram[new_img[x][y]] += 1

                # concatenazione
                histogram = np.concatenate((histogram, regionHistogram), axis=None)
        # normalizzazione
        cv2.normalize(histogram, histogram)
        return histogram


if __name__ == '__main__':
    db = Database.Database(False)
    data = db.get_normalized_template(0, db.gallery_data)
    lbp = Local_Binary_Pattern(1, 8, data)
    new_img = lbp.compute_lbp()
    hist = lbp.createHistogram(new_img)
    while True:
        cv2.imshow('frame', lbp.img.astype(np.uint8))
        cv2.imshow('new frame', np.array(new_img).astype(np.uint8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
