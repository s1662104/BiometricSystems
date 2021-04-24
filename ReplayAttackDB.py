import Main
import cv2

class Database:
    def __init__(self):
        self.directory = 'ReplayAttackDB'

    def main(self, real: bool):
        if real:
            dir = self.directory+"/Real/"
        else:
            dir = self.directory+"/Fake/"
        count=0
        while(True):
            crop = Main.videoCapture()
            cv2.imwrite(dir + str(count) + ".jpg", crop)
            count+=1

if __name__ == '__main__':
    Database().main(True)
