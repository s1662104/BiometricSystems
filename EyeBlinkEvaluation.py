import csv
import os

import Antispoofing


class EyeBlinkEvaluation:
    def __init__(self):
        pass

    def writeEyeBlinkCsv (self,eyeblink,val):
        print(eyeblink)
        list = []
        for val_array in eyeblink:
            list.append(val_array)
        list.append(val)

        with open('eyeblink.csv', 'a+') as cvsfile:
            writer = csv.writer(cvsfile, delimiter=';')
            writer.writerow(list)

            # cvsfile.write(str(val))
            # cvsfile.write('\n')
            cvsfile.close()
    def createDataSetEyeBlink(self):
        root_dir = 'Data/EyeBlink/'
        current_real = 'Real/'
        current_fake = 'Fake/'

        ###Real Part
        src_real = root_dir + current_real

        realFileNames = os.listdir(src_real)

        realFileNames = [src_real + name for name in realFileNames]

        print('REAL')
        print('Total video Real: ', len(realFileNames))

        ###Fake Part
        src_fake = root_dir + current_fake

        fakeFileNames = os.listdir(src_fake)

        fakeFileNames = [src_fake + name for name in fakeFileNames]

        print('FAKE')
        print('Total video Fake: ', len(fakeFileNames))

        for name in realFileNames:
            list = []
            list.append(name)
            list.append(1)
            print(name)
            var = Antispoofing.EyeBlink(name).eyeBlinkStart()
            if var :
                val = 1
            else:
                val = 0

            self.writeEyeBlinkCsv(list, val)

        for name in fakeFileNames:
            list = []
            list.append(name)
            list.append(0)
            print(name)
            var = Antispoofing.EyeBlink(name).eyeBlinkStart()

            if var :
                val = 1
            else:
                val = 0

            self.writeEyeBlinkCsv(list, val)







if __name__ == '__main__':
    #EyeBlinkEvaluation().createDataSetEyeBlink()
    Antispoofing.EyeBlink(None).eyeBlinkStart()
