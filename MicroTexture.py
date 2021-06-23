import cv2
import pickle
import dlib
import AntiSpoofingTrainingEvaluation
import LBP
from MicroTextureSplitting import MicroTextureSplitting

dim_image = 64


# funzione che inquadra l'utente e ritaglia il volto. Si utilizzano i landmark del volto per un'acquisizione
# piu' precisa
def detect_face(img, vis, crop=None):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        landmark = predictor(img, d)
        top = landmark.part(19).y
        left = landmark.part(0).x
        right = landmark.part(16).x
        bottom = landmark.part(8).y
        crop = img[top:bottom, left:right]
        cv2.rectangle(vis, (left, top), (right, bottom), (0, 255, 0), 3)
        try:
            crop = cv2.resize(crop, (dim_image, dim_image))
        except Exception as e:
            print(str(e))
    if len(dets) > 0:
        try:
            cv2.imshow('Face', crop)
        except Exception as e:
            print(str(e))
    return crop


class MicroTexture:
    # def __init__(self, nameFileCsv):
    #     self.nameFileCsv = nameFileCsv

    # Viene effettuata la verifica tramite webcam se abbiamo una persona reale, oppure abbiamo davanti alla webcam
    # un video/foto in esecuzione sul dispositivo dove la webcam sta puntando .
    def microTextureCam(self):

        cap = cv2.VideoCapture(0)
        val = False

        # andiamo a prendere un frame e lo convertiamo in scala di grigi.
        while True:
            ret, frame = cap.read()

            vis = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # effettuiamo il ritaglio del viso
            crop = detect_face(gray, vis)

            # qui effettuiamo la conversione dell'immagine croppata in LBP
            if crop is not None:
                myLBP = LBP.Local_Binary_Pattern(1, 8, crop)
            else:
                continue
            new_img = myLBP.compute_lbp()
            # creiamo l'histogram dell'immagine calcolata in lpb
            hist = myLBP.createHistogram(new_img)

            # Andiamo a prendere il classificatore già addestrato e salvato.
            with open('modelSVM.pkl', 'rb') as f:
                clf = pickle.load(f)
            hist = hist.reshape(1, -1)
            # attraverso il classificatore che abbiamo recuperato, ci facciamo dire se l'immagine è reale oppure no
            value = (clf.predict(hist))
            print(value)
            if value == 0:
                print("REAL")
                val = True
                break
            else:
                print("FAKE")
                val = False
                break
        cap.release()
        cv2.destroyAllWindows()
        return val

    def microTextureVideo(self, pathVid):
        cap = cv2.VideoCapture(pathVid)
        val = False

        # andiamo a prendere un frame e lo convertiamo in scala di grigi.
        while True:
            ret, frame = cap.read()
            try:
                vis = frame.copy()
            except Exception as e:
                print(str(e))
                break

            # convertiamo in scala di grigi.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # effettuiamo il ritaglio del viso
            crop = detect_face(gray, vis)

            # qui effettuiamo la conversione dell'immagine croppata in LBP
            if crop is not None:
                myLBP = LBP.Local_Binary_Pattern(1, 8, crop)
            else:
                continue
            new_img = myLBP.compute_lbp()
            # creiamo l'histogram dell'immagine calcolata in lpb
            hist = myLBP.createHistogram(new_img)

            # Andiamo a prendere il modello trained e salvato.
            with open('modelSVM.pkl', 'rb') as f:
                clf = pickle.load(f)

            hist = hist.reshape(1, -1)
            # attraverso il classificatore che abbiamo recuperato, ci facciamo dire se l'immagine è reale oppure no
            value = (clf.predict(hist))
            print(value)
            if value == 0:
                print("REAL")
                val = True
                break
            else:
                print("FAKE")
                val = False
                break

        cap.release()
        cv2.destroyAllWindows()
        return val

    # viene effettuata l'evaluation dal file csv, nel caso in cui questa funzione viene richiamata da replayAttackCam,
    # non mostra i calcoli e grafici
    def microTextureEvaluation(self, nameFileCsv):
        X_train, X_test, y_train, y_test = MicroTextureSplitting(nameFileCsv).splitting_train_test()


        # viene restituito il modello SVM addestrato e i valori predetti.
        svm, y_test_score = AntiSpoofingTrainingEvaluation.ModelSVM(X_train, y_train, X_test,
                                                                                    y_test).train_svm()

        # viene scritto il modello su file
        with open('modelSVM.pkl', 'wb') as f:
            pickle.dump(svm, f)

        # viene stampata la roc-curve e i valori FRR e SFAR.
        AntiSpoofingTrainingEvaluation.plot_roc_curve(y_test, y_test_score)
        FRR, SFAR = AntiSpoofingTrainingEvaluation.spoofing_scenario(y_test, y_test_score, index=1)
        print("#######################")
        print("###Spoofing Scenario###")
        print("#######################")
        print("FRR: ", FRR)
        print("SFAR: ", SFAR)

        return svm


def main():
    nameFileCsv = 'histogram.csv'
    # MicroTexture(nameFileCsv).microTextureCam()
    MicroTexture().microTextureEvaluation(nameFileCsv)


if __name__ == '__main__':
    main()
