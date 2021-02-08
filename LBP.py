import Database
import cv2
import numpy as np
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


    # def define_window(self):
    #     neighbor = self.radius*8
    #     count = int(neighbor/self.neighborhood)-1
    #     min = -self.radius
    #     max = self.radius+1
    #     str = ""
    #     c = count
    #     print(count, c)
    #     for x in range(min, max):
    #         for y in range(min, max):
    #             if abs(x)==self.radius or abs(y)==self.radius:
    #                 if c==0:
    #                     str = str+"1"
    #                 else: str = str+"0"
    #                 c = c-1
    #             else: str = str+"0"
    #             if c<0:
    #                 c = count
    #         str = str+"\n"
    #     print(str)

    def define_window(self):
        dim = self.radius*2+1
        x = np.ones((dim, dim))
        x[1:-1, 1:-1] = 0
        print(x)

    def find_pixels(self, cx, cy):
        #dividere un angolo di 360 in self.neighborhood parti
        angles_array = 2*np.pi/self.neighborhood
        #ottenere tutti gli angoli
        alpha = np.arange(0, 2 * np.pi, angles_array)
        #calcolare coppia di seno e coseno per ogni angolo
        s_points = np.array([-np.sin(alpha), np.cos(alpha)]).transpose()
        s_points *= self.radius
        print(s_points)
        print(self.img[cx][cy])

        # for i in range(s_points.shape[0]):
        #     print(i,cx,cy)




if __name__ == '__main__':
    # db = Database.Database()
    # data = db.get_normalized_template(1)

    lbp = LBP(1,8, None)
    print(lbp.img)
    lbp.find_pixels(4,4)

    # while(True):
    #     cv2.imshow('frame', data)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

