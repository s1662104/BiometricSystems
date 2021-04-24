import Main
import cv2

class Database:
    def __init__(self):
        self.directory = 'ReplayAttackDB'

    def main(self, real: bool, c=None):
        if real:
            dir = self.directory+"/Real/"
        else:
            dir = self.directory+"/Fake/"
        if c is None:
            count=0
        else:
            count = c
        while(True):
            crop = Main.videoCapture()
            cv2.imwrite(dir + str(count) + ".jpg", crop)
            count+=1

if __name__ == '__main__':
    Database().main(False,0)
