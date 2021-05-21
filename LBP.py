import dlib
import numpy as np
import cv2
import Database
from sklearn.svm import SVC

class Local_Binary_Pattern:

    def __init__(self, radius, neighborhood, img):
        self.radius = radius
        self.neighborhood = neighborhood
        if self.radius <= 0 or self.neighborhood < 4 or self.neighborhood > self.radius*8:
            raise Exception("Input error")
        self.img = img

    def compute_lbp(self):
        new_img = [[0 for y in range(self.img.shape[0])] for x in range(self.img.shape[1])]
        matrix = []
        count=-1
        for i in range(10-self.radius, 10+self.radius+1):
            matrix.append([])
            count+=1
            for j in range(10-self.radius, 10+self.radius+1):
                matrix[count].append(self.img[i][j])
        print(matrix)
        pixels = self.find_neighbors(10, 10)
        # for i in range(0, self.img.shape[0]):
        #     for j in range(0, self.img.shape[1]):
        #         # print("----------------")
        #         # print("i;j:", i, j)
        #         # print("value:", self.img[i][j])
        #         # print("neighborhood:")
        #         # print(self.img[i - self.radius: i + self.radius + 1, j - self.radius: j + self.radius + 1])
        #         pixels = self.find_neighbors(i, j)
        #         # print("PIXELS",pixels)
        #         pattern = np.where(pixels > self.img[i][j], 1, 0)
        #         # print("PATTERN:",pattern)
        #         value = 0
        #         count = 0
        #         for k in pattern:
        #             value += k * 2 ** count
        #             count += 1
        #         new_img[i - self.radius][j - self.radius] = value % 256
        #         # print("new value:", value)
        #         # print("----------------")
        # # print("NEW IMG", new_img)
        return new_img

    def find_neighbors(self, cx, cy):
        # definire il cerchio composto da self.neighborhood punti
        angles_array = 2*np.pi/self.neighborhood
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
            # print(x,y)
            if x_fract == 0 and y_fract == 0:
                coorx = int(x)
                coory = int(y)
                if self.check_border(cx+coorx,cy+coory):
                    pixels.append(0)
                else:
                    pixels.append(self.img[cx+coorx][cy+coory])
            else:
                x_c = np.ceil(x).astype(int)
                y_c = np.ceil(y).astype(int)
                x_f = np.floor(x).astype(int)
                y_f = np.floor(y).astype(int)
                print(x,y)
                print(x_c,y_c)
                print(x_f,y_f)
                print("--------------")
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
                if self.check_border(cx+x1,cy+y1) or self.check_border(cx+x2,cy+y2):
                    pixels.append(0)
                else:
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

    def check_border(self,x,y):
        return x < 0 or x >= self.img.shape[0] or y < 0 or y >= self.img.shape[1]

    def createHistogram(self, new_img,grid_x = 8,grid_y = 8):
        histogram = []

        #Check the pixels  matrix
        if len(new_img) == 0:
            raise Exception("Input error")

        #Get the matrix dimensions
        h = len(new_img)
        w = len(new_img[0])

        #The LBPHFaceRecognizer uses Extended Local Binary Patterns
        #(it's probably configurable with other operators at a later
        #point), and has the following default values for radius = 1 and neighbors = 8
        # grid_x = 8
        # grid_y = 8

        #Get the size (width and height) of each region
        gridWidth = int(w / grid_x)
        gridHeight = int(h / grid_y)

        #Calculates the histogram of each grid
        for gx in range(0, grid_x):
            for gy in range(0, grid_y):

                #Create a slice with empty 256 positions
                regionHistogram = [0]*256

                #Define the start and end positions for the following loop
                startPosX = gx * gridWidth
                startPosY = gy * gridHeight
                endPosX = (gx+1) * gridWidth
                endPosY = (gy+1) * gridHeight

                #Creates the histogram for the current region
                for x in range(startPosX, endPosX):
                    for y in range(startPosY, endPosY):
                        if new_img[x][y] < len(regionHistogram):
                            regionHistogram[new_img[x][y]] += 1

                #Concatenate two slices
                histogram = np.concatenate((histogram, regionHistogram), axis=None)
        cv2.normalize(histogram, histogram)
        return histogram


class Spoof_Local_Binary_Pattern(Local_Binary_Pattern):

    def createHistogram(self, new_img, grid_x=8, grid_y=8):
        w = len(new_img)
        h = len(new_img[0])
        # dividiamo la nostra immagine in 4 x 4 blocchi
        grid_x = int(w / 16)
        grid_y = int(h / 16)
        return super().createHistogram(new_img, grid_x, grid_y)

if __name__ == '__main__':
    db = Database.Database()
    data = db.gallery_data[0]
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    dets = detector(data, 1)
    for i, d in enumerate(dets):
       crop = data[d.top() : d.bottom(), d.left() : d.right()]
       crop = cv2.resize(crop, (64, 64))
    print(crop)
    lbp = Local_Binary_Pattern(2, 16, crop)
    new_img = lbp.compute_lbp()
    # hist = lbp.createHistogram(new_img)
    # while True:
    #     cv2.imshow('frame', lbp.img.astype(np.uint8))
    #     cv2.imshow('new frame', np.array(new_img).astype(np.uint8))
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
