import numpy as np

class Local_Binary_Pattern:

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

    def compute_lbp(self):
        new_img = [[0 for y in range(self.img.shape[0]-2)] for x in range(self.img.shape[1]-2)]
        pixels = lbp.find_neighbors(4, 4)
        print("PIXELS",pixels)
        pattern = np.where(pixels > lbp.img[4][4],1,0)
        print("PATTERN:",pattern)
        value=0
        count=0
        for i in pattern:
            value += i * 2**count
            count+=1
        new_img[4][4]=value
        print(value)
        print("NEW IMG", new_img)



    def find_neighbors(self, cx, cy):
        #dividere un angolo di 360 in self.neighborhood parti
        angles_array = 2*np.pi/self.neighborhood
        #ottenere tutti gli angoli
        alpha = np.arange(0, 2 * np.pi, angles_array)
        #ordiniamo i gradi in modo tale da partire dall'angolo in alto a sx e procedere verso dx
        alpha = lbp.sort_points(alpha)
        print(np.degrees(alpha))
        #calcolare coppia di seno e coseno per ogni angolo
        s_points = np.array([-np.sin(alpha), np.cos(alpha)]).transpose()
        s_points *= self.radius
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
            #print(x,y)
            if (x_fract==0 and y_fract==0):
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
                value = self.bilinear_interpolation(x1,y1,x2,y2,cx,cy,x,y)
                pixels.append(np.round(value).astype(int))
        return pixels

    def bilinear_interpolation(self,x1,y1,x2,y2,cx,cy,x,y):
        Q11 = self.img[cx + x1][cy + y1]
        Q21 = self.img[cx + x1][cy + y2]
        Q12 = self.img[cx + x2][cy + y1]
        Q22 = self.img[cx + x2][cy + y2]
        dem = (x2 - x1) * (y2 - y1)
        return (Q11*(x2-x)*(y2-y)/dem)+(Q21*(x-x1)*(y2-y)/dem)+\
                (Q12 * (x2 - x) * (y - y1) / dem)+(Q22 * (x - x1) * (y - y1) / dem)

    def sort_points(self,alpha):
        if self.neighborhood>4:
            left_point=np.where(np.degrees(alpha)==135.)[0][0];
        else:
            left_point=np.where(np.degrees(alpha)==90.)[0][0];
        new_alpha = []
        count = left_point
        while len(new_alpha)!=self.neighborhood:
            new_alpha.append(alpha[count])
            if np.degrees(alpha[count])==0.:
                count = self.neighborhood
            count -=1
        return new_alpha

if __name__ == '__main__':
    # db = Database.Database()
    # data = db.get_normalized_template(1)
    lbp = Local_Binary_Pattern(1,8, None)
    print(lbp.img)
    print(lbp.img[4 - lbp.radius: 4 + lbp.radius + 1, 4 - lbp.radius: 4 + lbp.radius + 1])
    lbp.compute_lbp()

    # while(True):
    #     cv2.imshow('frame', data)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

