import Database
import cv2
import numpy as np

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

    def find_neighborhood(self):
        x = np.arange(0,self.neighborhood)
        x = self.radius * np.cos(2 * np.pi * x / self.neighborhood)

        y = np.arange(0, self.neighborhood)
        y = -self.radius * np.sin(2 * np.pi * x / self.neighborhood)
        print(x,y)



if __name__ == '__main__':
    #db = Database.Database()
    #data = db.get_normalized_template(1)

    lbp = LBP(1,8, None)
    lbp.find_neighborhood()

    # while(True):
    #     cv2.imshow('frame', data)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

