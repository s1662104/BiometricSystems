#######################################################
# Qui abbiamo la classe EyeBlink il cui compito è quello
# di rilevare se uno utente tramite video o tramite
# webcam sta battendo le palpebre oppure no.
#######################################################

import cv2
import imutils
from imutils.video import FileVideoStream
from imutils.video import FPS
import time
import numpy as np
from scipy.spatial import distance

from imutils import face_utils
import dlib

CONSEC_FRAMES_NUMBER = 3


# TODO cos'e' inputType?
class EyeBlink:
    def __init__(self, inputType):
        # inputType è la stringa che contiene il path per un video, oppure non contiene nulla.
        self.inputType = inputType

    # Il metodo sottostante va a prendere un frame per volta dal video (se è stato passato un input alla classe) oppure
    # utilizza come sorgente la webcam e passa i frame ai relativi metodi "eye_blink_video"(se si tratta di un video)
    # o "eye_blink_cam" se la sorgente è una webcam, tali metodi andranno ad analizzare i singoli frame e ritornano
    # dei parametri come l'EAR_TOP ( che è Eye_aspect_ratio più alto che si incontra durante l'analisi dei frame
    # di uno streaming video). Durante il conteggio dei blinking se si verifica almeno un eye-blink
    # ritorna True, altrimenti continua e se non è stato mai rilevato un eye-blinking torna False.
    # In questo caso viene utilizzato un threshold adattivo
    def eyeBlinkStart(self):
        # inizializzazione di counter e total, counter conta i frame consecutivi in EAR < threshold, total il numero di
        # eyeblink
        COUNTER = 0
        TOTAL = 0

        # se inputType contiene un path per un video allora...
        if self.inputType is not None:

            ear_top = 0

            history = ' '
            print(self.inputType)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            # TODO scrivere i commenti tutti in italiano, da' l'idea che il codice non e' scritto da noi e non e' stato nemmeno compreso
            # start the file video stream thread and allow the buffer to
            # start to fill
            fvs = FileVideoStream(self.inputType).start()
            time.sleep(1.0)

            # start the FPS timer
            fps = FPS().start()

            # fileStream = True
            num_frames = 0
            # TODO spiegare perche' 150 frames
            # loop over frames from the video file stream until 150 frames
            while fvs.more() and num_frames < 150:

                frame = fvs.read()
                # resize frame with width = 300
                try:
                    frame = imutils.resize(frame, width=300)
                except Exception as e:
                    print(str(e))

                # the frame is converted into grayscale image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # frame = np.dstack([frame, frame, frame])
                # viene richiamata la funzione che va ad analizzare ogni singolo frame per vedere se ci è stato
                # un eyeblink o meno
                try:
                    eyesdetect, COUNTER, TOTAL, ear_top = self.eye_blink_video(frame, detector, predictor,
                                                                               COUNTER, TOTAL, ear_top)
                except Exception as e:
                    print(str(e))
                    continue
                # TODO spiegare a cosa serve history, visto che non viene piu' usato
                # here we update the history
                history += eyesdetect
                # se si verifica un eyeblink ritorniamo True
                if TOTAL > 0:
                    cv2.destroyAllWindows()
                    fvs.stop()
                    return True

                # viene aggiornato fps e incrementato il numero di frame.
                fps.update()
                fps.stop()
                print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
                print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
                num_frames += 1

            # usciti dal while se non si è verificato Eyeblink ritorniamo False
            cv2.destroyAllWindows()
            fvs.stop()
            if TOTAL == 0:
                return False
        # se inputType è vuoto significa che usiamo la webcam
        elif self.inputType is None:
            # acquisiamo da webcam
            cap = cv2.VideoCapture(0)

            history = ''

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            ear_top = 0
            while True:

                ret, frame = cap.read()
                # viene richiamata la funzione che va ad analizzare ogni singolo per vedere se ci è stato
                # un eyeblink o meno
                eyedetect, TOTAL, COUNTER, ear_top = self.eye_blink_cam(self, frame, ret, detector, predictor,
                                                                        COUNTER, TOTAL, ear_top)
                # TODO spiegare a cosa serve history
                # here we update the history
                history += eyedetect
                # se total è maggiore di 0 significa che un eyeblink è avvenuto e ritorniamo true
                if TOTAL >= 1:
                    cv2.putText(frame, "Real", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                    cap.release()
                    cv2.destroyAllWindows()
                    print("EYEBLINK: REAL")
                    return True
                # se ciò non avviene per 200 frame ritorniamo false
                elif len(history) > 200:
                    print(history)
                    result = self.isBlinking(history, 3)
                    print(result)

                    cv2.putText(frame, "Fake", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    cap.release()
                    cv2.destroyAllWindows()
                    print("EYEBLINK: FAKE")
                    return False
                else:
                    cv2.putText(frame, "Checking...", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 3,
                                cv2.LINE_AA)

            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

            # if the `q` key was pressed, break from the loop
        cap.release()
        cv2.destroyAllWindows()

    # Il metodo sottostante value l'eye_aspect_ratio in questo modo:
    # EAR = (||p2 - p6||+ ||p3 - p5||)/(2||p1-p4||)
    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        EAR = (A + B) / (2.0 * C)
        return EAR

    # metodo secondario che serve per vedere se si è verificato un blinking
    # TODO serve? commentare
    def isBlinking(self, history, maxFrames):
        for i in range(maxFrames):
            pattern = '1' + '0' * (i + 1) + '1'
            if pattern in history:
                return True
        return False

        # gestisce gli eyeblink della cam

    def eye_blink_cam(self, frame, rect, detector, predictor, COUNTER, TOTAL, ear_top):
        eyes_detect = ''

        # resize del frame a width a 450
        frame = imutils.resize(frame, width=450)
        # conversione in scala di grigi
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)  # Detect the faces in the image
        (left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # andiamo a rilevare il rettangolo del volto e croppiamo l'immagine con 250x250
        for det in dets:
            (x, y, w, h) = face_utils.rect_to_bb(det)
            crop = gray[y:y + h, x:x + w]
            try:
                crop = cv2.resize(crop, (250, 250))
            except Exception as e:
                print(str(e))
                break

            dets1 = detector(crop, 1)

            # rilevamento degli occhi e calcolo dell'EAR di entrambe gli occhi
            for det in dets1:
                shape = predictor(crop, det)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[left_s:left_e]
                rightEye = shape[right_s:right_e]
                left_eye_EAR = self.eye_aspect_ratio(leftEye)
                right_eye_EAR = self.eye_aspect_ratio(rightEye)
                ear = (left_eye_EAR + right_eye_EAR) / 2.0

                # se ear_top è già stato assegnato allora viene calcolato il threshold
                if ear_top != 0:
                    ear_threshold = (ear_top * 2) / 3
                    print("Ear_th", ear_threshold)
                    print("EAR TOP", ear_top)

                    # se ear < del suo threshold
                    if ear < ear_threshold:
                        # abbiamo lo stato di occhi chiusi e incrementiamo il contatore dei frame
                        eyes_detect = '1'
                        COUNTER += 1
                    else:
                        # quando l'occhio è aperto o di nuovo aperto andiamo a confrontare il contatore
                        # se ha raggiunto i minimi frame consecutivi e se lo è abbiamo avuto un eyeblink
                        # di conseguenza il contatore viene azzerato.
                        eyes_detect = '0'

                        if COUNTER >= CONSEC_FRAMES_NUMBER:
                            TOTAL += 1

                        COUNTER = 0

                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # ear è maggiore di ear_top lo sostituisce, ricordo che ear_top ad inizio video è a 0 e serve per calcolare il threshold
                if ear > ear_top:
                    ear_top = ear

        return eyes_detect, TOTAL, COUNTER, ear_top

    # gestisce gli eyeblink dei video
    def eye_blink_video(self, frame, detector, predictor, COUNTER, TOTAL, ear_top):
        eyes_detect = ''

        rects = detector(frame, 1)

        (left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # andiamo a rilevare il rettangolo del volto e croppiamo l'immagine con 200x200
        # TODO perche' sopra il crop e' 250 e qui 200?
        for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            crop = frame[y:y + h, x:x + w]
            try:
                crop = cv2.resize(crop, (200, 200))
            except Exception as e:
                print(str(e))
                break

            rects1 = detector(crop, 1)

            for rect in rects1:
                shape = predictor(crop, rect)
                shape = face_utils.shape_to_np(shape)

                # rilevamento degli occhi e calcolo dell'EAR di entrambe gli occhi
                leftEye = shape[left_s:left_e]
                rightEye = shape[right_s:right_e]
                left_eye_EAR = self.eye_aspect_ratio(leftEye)
                right_eye_EAR = self.eye_aspect_ratio(rightEye)
                ear = (left_eye_EAR + right_eye_EAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                # se ear_top è già stato assegnato allora viene calcolato il threshold
                if ear_top != 0:
                    ear_threshold = (ear_top * 2) / 3
                    print("Ear_th", ear_threshold)
                    print("EAR TOP", ear_top)

                    # se ear < del suo threshold
                    if ear < ear_threshold:
                        # abbiamo lo stato di occhi chiusi e incrementiamo il contatore dei frame
                        eyes_detect = '1'
                        COUNTER += 1
                    else:
                        # quando l'occhio è aperto o di nuovo aperto andiamo a confrontare il contatore
                        # se ha raggiunto i minimi frame consecutivi e se lo è abbiamo avuto un eyeblink
                        # e aumentiamo total (indica il numero di eyeblink) di conseguenza il contatore viene azzerato.
                        eyes_detect = '0'
                        if COUNTER >= CONSEC_FRAMES_NUMBER:
                            TOTAL += 1

                        COUNTER = 0

                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ear è maggiore di ear_top lo sostituisce, ricordo che ear_top ad inizio video è a 0 e serve
                # per calcolare il threshold
                if ear > ear_top:
                    ear_top = ear

            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
            return eyes_detect, COUNTER, TOTAL, ear_top

    # TODO spiegare perche' non e' stato analizzato tutto il video !!!
    # Il metodo sottostante viene utilizzato con i vari thresholds fissi variabili.
    # Si va a prendere un frame per volta dal video; e ciascun frame viene passato al metodo "eye_blink_video_fixedTh"
    # che andrà ad analizzare i singoli frame, confrontando l'eye_aspect_ratio del frame corrente con i vari threshold.
    # In questo caso si analizzano un certo numero di frame e non tutto il video e viene, infine, tornata la lista dei
    # valori che abbiamo ottenuto in base al threshold: se EAR < 'valore_del_threshold_x' avremo th_x = 1
    # in corrispondenza del threshold 'x' nella lista.
    def eyeBlinkStartThFixed(self):
        # inizializzazione di counter e total, counter conta i frame consecutivi in EAR < threshold, total il numero di
        # eyeblink
        COUNTER = 0
        TOTAL = 0
        # dichiarazione e inizializzazione dei threshold che vanno da 0.10 a 0.29
        ear_th = []
        for threshold in np.arange(0.10, 0.30, 0.01):
            ear_th.append(0)

        print(self.inputType)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # inputType ha il suo interno il path + il video
        # poi inizia il video stream.
        fvs = FileVideoStream(self.inputType).start()
        time.sleep(1.0)
        # TODO commentare tutto in italiano, vedi sopra per i motivi
        # start the FPS timer
        fps = FPS().start()

        num_frames = 0
        # TODO spiegare nel report perche' bastano 150 frames
        # loop over frames from the video file stream until 150 frames (sono sufficienti per verificare
        # se è avvenuto un eyeblink o meno)
        while fvs.more() and num_frames < 150:

            frame = fvs.read()

            try:
                frame = imutils.resize(frame, width=300)
            except Exception as e:
                print(str(e))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # frame = np.dstack([frame, frame, frame])

            # viene richiamata la funzione che va ad analizzare ogni singolo frame per vedere se ci è stato
            # un eyeblink o meno
            try:

                eyesdetect, COUNTER, ear_th = self.eye_blink_video_fixedTh(frame, detector,
                                                                           predictor, COUNTER, ear_th)
            except Exception as e:
                print(str(e))
                continue

            # viene aggiornato fps e incrementato il numero di frame.
            fps.update()
            fps.stop()
            print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
            num_frames += 1

        cv2.destroyAllWindows()
        fvs.stop()
        # ritorna la lista dei valori relativi ai threshold
        return ear_th

    def eye_blink_video_fixedTh(self, frame, detector, predictor, COUNTER, TOTAL, ear_th):

        eyes_detect = ''

        rects = detector(frame, 1)  # Detect the faces in the image
        (left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # TODO nuovamente, perche' in alcuni casi 250x250 e in altri 200x200? in questi casi si usano delle costanti
        # andiamo a rilevare il rettangolo del volto e croppiamo l'immagine con 200x200
        for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            crop = frame[y:y + h, x:x + w]

            try:
                crop = cv2.resize(crop, (200, 200))
            except Exception as e:
                print(str(e))
                break

            rects1 = detector(crop, 1)

            for rect in rects1:
                shape = predictor(crop, rect)
                shape = face_utils.shape_to_np(shape)

                # rilevamento degli occhi e calcolo dell'EAR di entrambe gli occhi
                leftEye = shape[left_s:left_e]
                rightEye = shape[right_s:right_e]
                left_eye_EAR = self.eye_aspect_ratio(leftEye)
                right_eye_EAR = self.eye_aspect_ratio(rightEye)
                ear = (left_eye_EAR + right_eye_EAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # confrontiamo EAR con i threshold che vanno da 0.10 a 0.29 e mettiamo 1 al relavito threshold se EAR è
                # minore di esso altrimenti il valore resta a 0.
                count = 0

                for threshold in np.arange(0.10, 0.30, 0.01):
                    # Fix the threshold
                    th = np.round(threshold, 2)
                    if ear < th:
                        COUNTER += 1
                        eyes_detect = '1'
                        ear_th[count] = 1
                    else:
                        eyes_detect = '0'

                        COUNTER = 0
                    count += 1
                    # print(count)

                    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (200, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # TODO questo serve? Perche' in teoria sono fissi no? quindi ear_top non serve
                # if ear > ear_top:
                #     ear_top = ear

            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
            return eyes_detect, COUNTER, ear_th


def main():
    EyeBlink(None).eyeBlinkStart()


if __name__ == '__main__':
    main()
