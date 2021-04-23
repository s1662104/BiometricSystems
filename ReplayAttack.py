import cv2
import imutils
import pickle


import dlib
import AntiSpoofingTrainingEvaluation
#import Main
import LBP
from ReplayAttackSplitting import ReplayAttackSplitting
dim_image = 64
def detect_face(img, vis, crop=None):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    dets = detector(img, 1)  # Detect the faces in the image
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


class ReplayAttack:
    def __init__(self,nameFileCsv):
        self.nameFileCsv = nameFileCsv

    #Viene effettuata la verifica tramite webcam se abbiamo una persona reale, oppure abbiamo davanti alla webcam
    # un video/foto in esecuzione sul dispositivo dove la webcam sta puntando .
    def replayAttackCam(self,nameFileCsv):
        cap = cv2.VideoCapture(0)

        while (True):
            crop = None
            ret, frame = cap.read()

            vis = frame.copy()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            crop = detect_face(gray, vis)

            if crop is not None:
                myLBP = LBP.Spoof_Local_Binary_Pattern(1, 8, crop)

            new_img = myLBP.compute_lbp()
            hist = myLBP.createHistogram(new_img)

            # Andiamo a prendere il modello trained e salvato.
            with open('modelSVM.pkl', 'rb') as f:
                clf = pickle.load(f)
            # nsamples = hist.shape
            # print("nsamples",nsamples)
            hist = hist.reshape(1, -1)
            # print(hist)
            value = (clf.predict(hist))
            print(value)
            if value == 0:
                print("REAL")
                return True
            else:
                print("FAKE")
                return False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # if the `q` key was pressed, break from the loop
        cap.release()
        cv2.destroyAllWindows()

    # viene effettuata l'evaluation dal file csv, nel caso in cui questa funzione viene richiamata da replayAttackCam,
    # non mostra i calcoli e grafici
    def replayAttackEvaluation(self,nameFileCsv):
        X_train, X_test, y_train, y_test = ReplayAttackSplitting(nameFileCsv).splitting_train_test(nameFileCsv)
        svm, y_train_score, y_test_score = AntiSpoofingTrainingEvaluation.ModelSVM(X_train, y_train, X_test,
                                                                                   y_test).train_svm()
        with open('modelSVM.pkl', 'wb') as f:
            pickle.dump(svm,f)

        AntiSpoofingTrainingEvaluation.plot_roc_curve(y_test, y_test_score)
        FRR, SFAR = AntiSpoofingTrainingEvaluation.spoofing_scenario(y_test, y_test_score, index=1)
        print("#######################")
        print("###Spoofing Scenario###")
        print("#######################")
        print("FRR: ", FRR)
        print("SFAR: ", SFAR)
        FRR, FAR, HTER = AntiSpoofingTrainingEvaluation.licit_scenario(y_test, y_test_score, index=1)
        print()
        print("####################")
        print("###Licit Scenario###")
        print("####################")
        print("FRR: ", FRR)
        print("FAR: ", FAR)
        print("HTER: ", HTER)

        return svm


def main():
    nameFileCsv = 'histogram.csv'
    #ReplayAttack(nameFileCsv).replayAttackCam(nameFileCsv)
    ReplayAttack(nameFileCsv).replayAttackEvaluation(nameFileCsv)



if __name__ == '__main__':
    main()