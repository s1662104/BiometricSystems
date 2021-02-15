import dlib
import numpy as np
import cv2
import Database

class Local_Binary_Pattern:

    def __init__(self, radius, neighborhood, img):
        self.radius = radius
        self.neighborhood = neighborhood
        if self.radius <= 0 or self.neighborhood < 4 or self.neighborhood > self.radius*8:
            raise Exception("Input error")
        self.img = img
        # if img == None:
        #     self.img = []
        #     for i in range(10):
        #         self.img.append(np.arange(10 * i, 10 * (i + 1)))
        #     self.img = np.array(self.img).astype(np.uint8)
        #     self.img = np.array([[10, 2, 32, 18, 81], [4, 73, 21, 10, 42], [54, 21, 17, 62, 49],
        #     [1, 72, 8, 92, 62], [7, 77, 28, 10, 88], ], np.int32)

    def compute_lbp(self):
        new_img = [[0 for y in range(self.img.shape[0] - 2)] for x in range(self.img.shape[1] - 2)]
        for i in range(self.radius, self.img.shape[0]-self.radius):
            for j in range(1, self.img.shape[1]-self.radius):
                # print("----------------")
                # print("i;j:", i, j)
                # print("value:", self.img[i][j])
                # print("neighborhood:")
                # print(lbp.img[i - lbp.radius: i + lbp.radius + 1, j - lbp.radius: j + lbp.radius + 1])
                pixels = lbp.find_neighbors(i, j)
                # print("PIXELS",pixels)
                pattern = np.where(pixels > lbp.img[i][j], 1, 0)
                # print("PATTERN:",pattern)
                value = 0
                count = 0
                for k in pattern:
                    value += k * 2 ** count
                    count += 1
                new_img[i - 1][j - 1] = value % 256
                # print("new value:", value)
                # print("----------------")
        # print("NEW IMG", new_img)
        return new_img

    def find_neighbors(self, cx, cy):
        # dividere un angolo di 360 in self.neighborhood parti
        angles_array = 2*np.pi/self.neighborhood
        # ottenere tutti gli angoli
        alpha = np.arange(0, 2 * np.pi, angles_array)
        # ordiniamo i gradi in modo tale da partire dall'angolo in alto a sx e procedere verso dx
        alpha = lbp.sort_points(alpha)
        # print(np.degrees(alpha))
        # calcolare coppia di seno e coseno per ogni angolo
        s_points = np.array([-np.sin(alpha), np.cos(alpha)]).transpose()
        s_points *= self.radius
        # print(self.img[cx][cy])
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
            # print(x,y)
            if x_fract == 0 and y_fract == 0:
                coorx = int(x)
                coory = int(y)
                pixels.append(self.img[cx+coorx][cy+coory])
            else:
                x_c = np.ceil(x).astype(int)
                y_c = np.ceil(y).astype(int)
                x_f = np.floor(x).astype(int)
                y_f = np.floor(y).astype(int)
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
                value = self.bilinear_interpolation(x1, y1, x2, y2, cx, cy, x, y)
                pixels.append(np.round(value).astype(int))
        return pixels

    def bilinear_interpolation(self, x1, y1, x2, y2, cx, cy, x, y):
        Q11 = self.img[cx + x1][cy + y1]
        Q21 = self.img[cx + x1][cy + y2]
        Q12 = self.img[cx + x2][cy + y1]
        Q22 = self.img[cx + x2][cy + y2]
        dem = (x2 - x1) * (y2 - y1)
        return (Q11*(x2-x)*(y2-y)/dem)+(Q21*(x-x1)*(y2-y)/dem) + \
            (Q12 * (x2 - x) * (y - y1) / dem)+(Q22 * (x - x1) * (y - y1) / dem)

    def sort_points(self, alpha):
        if self.neighborhood > 4:
            left_point = np.where(np.degrees(alpha) == 135.)[0][0]
        else:
            left_point = np.where(np.degrees(alpha) == 90.)[0][0]
        new_alpha = []
        count = left_point
        while len(new_alpha) != self.neighborhood:
            new_alpha.append(alpha[count])
            if np.degrees(alpha[count]) == 0.:
                count = self.neighborhood
            count -= 1
        return new_alpha

if __name__ == '__main__':
    db_index = input("Quale database si vuole utilizzare? \n 0 - Olivetti \n 1 - LFW\n")
    if db_index == "0":
        db = Database.Database(0)
        data = db.get_normalized_template(0)
        lbp = Local_Binary_Pattern(1, 8, data)
        print(lbp.img)
        new_img = lbp.compute_lbp()
        while (True):
            cv2.imshow('frame', data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif db_index == "1":
        db = Database.Database(1)
        data = db.get_normalized_template(0)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        dets = detector(data, 1)
        for i, d in enumerate(dets):
           crop = data[d.top() : d.bottom(), d.left() : d.right()]
           crop = cv2.resize(crop, (64, 64))

    while True:
        cv2.imshow('frame', lbp.img.astype(np.uint8))
        cv2.imshow('new frame', np.array(new_img).astype(np.uint8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        while (True):
            cv2.imshow('frame', crop)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("Valora non valido!")

