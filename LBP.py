import Database
import cv2
import numpy as np
from scipy import interpolate
import math

class LBP:

    def __init__(self, radius, neighborhood, img):
        self.radius = radius
        self.neighborhood = neighborhood
        if self.radius <= 0 or self.neighborhood < 4 or self.neighborhood>self.radius*8:
            raise Exception("Input error")
        if (img==None):
            self.img = []
            for i in range(10):
                self.img.append(np.arange(10 * i, 10 * (i + 1)))
            self.img = np.array(self.img).astype(np.uint8)


    def define_window(self):
        dim = self.radius*2+1
        x = np.ones((dim, dim))
        x[1:-1, 1:-1] = 0
        print(x)

    def find_neighbors(self, cx, cy):
        #dividere un angolo di 360 in self.neighborhood parti
        angles_array = 2*np.pi/self.neighborhood
        #ottenere tutti gli angoli
        alpha = np.arange(0, 2 * np.pi, angles_array)
        #calcolare coppia di seno e coseno per ogni angolo
        s_points = np.array([-np.sin(alpha), np.cos(alpha)]).transpose()
        s_points *= self.radius
        print(s_points)
        print(self.img[cx][cy])
        # per ogni punto
        pixels = []
        for x,y in s_points:
            #analizziamo la parte frazionaria, considerando solo i primi quattro valori.
            # Se questi sono 0, allora si riferiscono ai punti nei gradi
            #0,90,180,270, che sono gli unici a trovarsi al centro del pixel.
            # In questo caso, non hanno bisogno di interpolazione
            x = np.round(x,4)
            y = np.round(y,4)
            x_fract = x - np.round(x)
            y_fract = y - np.round(y)
            if (x_fract==0 and y_fract==0):
                coorx = int(x)
                coory = int(y)
                pixels.append(self.img[cx+coorx][cy+coory])
            else:
                x_min = np.ceil(x).astype(int)
                y_min = np.ceil(y).astype(int)
                x_max = np.floor(x).astype(int)
                y_max = np.floor(y).astype(int)
                print("-----------")
                print(x,y)
                print(x_min,y_min)
                print(x_max,y_max)
                print(self.img[cx+x_min,cx+y_min],self.img[cx+x_max,cx+y_max],self.img[cx+x_min,cx+y_max],self.img[cx+x_max,cx+y_min])
        print(pixels)

if __name__ == '__main__':
    # db = Database.Database()
    # data = db.get_normalized_template(1)

    lbp = LBP(1,8, None)
    print(lbp.img)
    print(lbp.img[4 - lbp.radius: 4 + lbp.radius + 1, 4 - lbp.radius: 4 + lbp.radius + 1])

    lbp.find_neighbors(4,4)

    # while(True):
    #     cv2.imshow('frame', data)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

