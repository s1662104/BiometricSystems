# Classe usata per realizzare il DB realizzato da noi
import Pages
import cv2


class Database:
    def __init__(self):
        self.directory = 'MicroTextureDB'

    def main(self, real: bool, c=None):
        # definisce se si sta catturando un utente reale o non
        if real:
            dir = self.directory + "/Real/"
        else:
            dir = self.directory + "/Fake/"
        # count viene usato per riprendere il conteggio di immagini realizzate, in caso si spezzetti il l'operazione
        # in piu' iterazioni
        if c is None:
            count = 0
        else:
            count = c
        while True:
            crop = Pages.videoCapture()
            cv2.imwrite(dir + str(count) + ".jpg", crop)
            count += 1


if __name__ == '__main__':
    Database().main(False, 0)
