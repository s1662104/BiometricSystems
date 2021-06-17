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
        self.inputType = inputType

    # TODO riscrivere questo perche' non si capisce
    # Il metodo sottostante va a prendere un frame per volta dal video (se è stato passato un input alla classe) oppure
    # utilizza come sorgente la webcam e passa i frame ai relativi metodi "eye_blink_video"(se si tratta di un video)
    # o "eye_blink_cam" se la sorgente è una webcam, tali metodi andranno ad analizzare i singoli frame e ritornano
    # dei parametri come l'EAR_TOP ( che è Eye_aspect_ratio più alto che si incontra durante l'analisi dei frame
    # di uno streaming video). Durante il conteggio dei blinking se si verifica almeno un eye-blink
    # ritorna True, altrimenti continua e se non è stato mai rilevato un eye-blinking torna False.
    # In questo caso viene utilizzato un threshold adattivo
    # TODO COMMENTARE OGNI PASSAGGIO
    def eyeBlinkStart(self):
        # TODO SERVE?
        COUNTER = 0
        TOTAL = 0
        # TODO serve avere due if se poi si fanno piu' o meno le stesse cose?
        if self.inputType is not None:

            ear_top = 0

            history = ' '
            print(self.inputType)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

            fvs = FileVideoStream(self.inputType).start()
            time.sleep(1.0)
            fps = FPS().start()

            # TODO serve? e' sempre True
            fileStream = True
            num_frames = 0
            while fvs.more() and num_frames < 150:
                # TODO serve? mai usato
                crop = None
                frame = fvs.read()

                try:
                    frame = imutils.resize(frame, width=300)
                except Exception as e:
                    print(str(e))
                # TODO non serve, togliere
                try:
                    vis = frame.copy()

                except Exception as e:
                    print(str(e))
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # TODO ?
                frame = np.dstack([frame, frame, frame])
                # TODO a che serve counter se non si usa?
                try:
                    eyesdetect, COUNTER, TOTAL, ear_top = self.eye_blink_video(frame, detector, predictor,
                                                                               COUNTER, TOTAL, ear_top)
                except Exception as e:
                    print(str(e))
                    continue
                # TODO serve?
                history += eyesdetect
                if TOTAL > 0:
                    # TODO serve var?
                    var = True
                    cv2.destroyAllWindows()
                    fvs.stop()
                    return var
                else:
                    # TODO serve var? non si usa MA RITORNA MAI FALSE?? NON VEDO NESSUN RETURN FALSE
                    var = False
                # TODO ?
                fps.update()
                fps.stop()
                print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
                print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
                num_frames += 1

            cv2.destroyAllWindows()
            fvs.stop()

        elif self.inputType is None:
            cap = cv2.VideoCapture(0)
            # Write the label with this font
            history = ''
            # TODO da togliere, non si usa
            font = cv2.FONT_HERSHEY_SIMPLEX

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            ear_top = 0
            while True:

                ret, frame = cap.read()
                # TODO non si usa, da togliere
                vis = frame.copy()
                # TODO a che serve counter se non si usa?
                eyedetect, TOTAL, COUNTER, ear_top = self.eye_blink_cam(frame, ret, detector, predictor,
                                                                        COUNTER, TOTAL, ear_top)
                history += eyedetect
                if TOTAL >= 1:
                    cv2.putText(frame, "Real", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                    cap.release()
                    cv2.destroyAllWindows()
                    print("EYEBLINK: REAL")
                    return True
                # TODO perche' questo sopra non c'era? Serve?
                elif len(history) > 100:
                    print(history)
                    result = self.isBlinking(history, 3)
                    print(result)
                    # TODO ?
                    if result:
                        pass
                    else:
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
        else:
            exit("Il valore di inputType è errato")

    # Il metodo sottostante value l'eye_aspect_ratio in questo modo:
    # EAR = (||p2 - p6||+ ||p3 - p5||)/(2||p1-p4||)
    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        EAR = (A + B) / (2.0 * C)
        return EAR

    # ToDo non so se lasciare questa funzione superflua. TOGLIERE se superflua
    def isBlinking(self, history, maxFrames):
        for i in range(maxFrames):
            pattern = '1' + '0' * (i + 1) + '1'
            if pattern in history:
                return True
        return False

    # TODO commentare ogni passaggio
    def eye_blink_cam(self, frame, rect, detector, predictor, COUNTER, TOTAL, ear_top):

        eyes_detect = ''
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)  # Detect the faces in the image
        (left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        for det in dets:
            (x, y, w, h) = face_utils.rect_to_bb(det)
            crop = gray[y:y + h, x:x + w]
            try:
                crop = cv2.resize(crop, (250, 250))
            except Exception as e:
                print(str(e))
                break

            dets1 = detector(crop, 1)
            for det in dets1:
                shape = predictor(crop, det)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[left_s:left_e]
                rightEye = shape[right_s:right_e]
                left_eye_EAR = self.eye_aspect_ratio(leftEye)
                right_eye_EAR = self.eye_aspect_ratio(rightEye)
                ear = (left_eye_EAR + right_eye_EAR) / 2.0

                if ear_top != 0:
                    ear_threshold = (ear_top * 2) / 3
                    print("Ear_th", ear_threshold)
                    print("EAR TOP", ear_top)

                    if ear < ear_threshold:

                        eyes_detect = '1'
                        COUNTER += 1
                    else:
                        eyes_detect = '0'
                        # TODO o togliere o spiegare
                        if COUNTER >= CONSEC_FRAMES_NUMBER:
                            TOTAL += 1

                        COUNTER = 0

                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if ear > ear_top:
                    ear_top = ear

        return eyes_detect, TOTAL, COUNTER, ear_top

    # TODO commentare ogni commento. Non e' uguale alla funzione precedente?
    def eye_blink_video(self, frame, detector, predictor, COUNTER, TOTAL, ear_top):

        eyes_detect = ''

        rects = detector(frame, 1)
        (left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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

                leftEye = shape[left_s:left_e]
                rightEye = shape[right_s:right_e]
                left_eye_EAR = self.eye_aspect_ratio(leftEye)
                right_eye_EAR = self.eye_aspect_ratio(rightEye)
                ear = (left_eye_EAR + right_eye_EAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                if ear_top != 0:
                    ear_threshold = (ear_top * 2) / 3
                    print("Ear_th", ear_threshold)
                    print("EAR TOP", ear_top)
                    # TODO o togliere o spiegare
                    if ear < ear_threshold:
                        COUNTER += 1
                        print(COUNTER)
                        eyes_detect = '1'
                    else:
                        eyes_detect = '0'
                        if COUNTER >= CONSEC_FRAMES_NUMBER:
                            TOTAL += 1

                        COUNTER = 0

                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if ear > ear_top:
                    ear_top = ear

            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
            return eyes_detect, COUNTER, TOTAL, ear_top

    # TODO spiegare perche' non e' stato analizzato tutto il video
    # Il metodo sottostante viene utilizzato con i vari thresholds fissi variabili.
    # Si va a prendere un frame per volta dal video; e ciascun frame viene passato al metodo "eye_blink_video_fixedTh"
    # che andrà ad analizzare i singoli frame, confrontando l'eye_aspect_ratio del frame corrente con i vari threshold.
    # In questo caso si analizzano un certo numero di frame e non tutto il video e viene, infine, tornata la lista dei
    # valori che abbiamo ottenuto in base al threshold: se EAR < 'valore_del_threshold_x' avremo th_x = 1
    # in corrispondenza del threshold 'x' nella lista.
    # TODO commentare ogni passaggio
    def eyeBlinkStartThFixed(self):
        COUNTER = 0
        TOTAL = 0

        ear_th = []
        # TODO va da 0.10 a 0.29
        for threshold in np.arange(0.10, 0.30, 0.01):
            ear_th.append(0)
        # TODO serve? Non viene usato
        history = ' '
        print(self.inputType)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # TODO serve passargli self.inputType?? Non serve per distinguere tra video e cam??
        fvs = FileVideoStream(self.inputType).start()
        time.sleep(1.0)
        fps = FPS().start()
        # TODO questo non serve
        fileStream = True
        num_frames = 0
        while fvs.more() and num_frames < 150:
            # TODO non viene usato
            crop = None
            frame = fvs.read()

            try:
                frame = imutils.resize(frame, width=300)
            except Exception as e:
                print(str(e))
            # TODO questo non serve
            try:
                vis = frame.copy()

            except Exception as e:
                print(str(e))
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # TODO cosa fa?
            frame = np.dstack([frame, frame, frame])
            try:
                # TODO servono counter, _?
                eyesdetect, COUNTER, _, ear_th = self.eye_blink_video_fixedTh(frame, detector,
                                                                              predictor, COUNTER, TOTAL, ear_th)
            except Exception as e:
                print(str(e))
                continue

            history += eyesdetect
            fps.update()
            fps.stop()
            print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
            num_frames += 1

        cv2.destroyAllWindows()
        fvs.stop()
        return ear_th

    # TODO commentare ogni passaggio
    def eye_blink_video_fixedTh(self, frame, detector, predictor, COUNTER, TOTAL, ear_th):
        eyes_detect = ''

        rects = detector(frame, 1)  # Detect the faces in the image
        (left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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

                leftEye = shape[left_s:left_e]
                rightEye = shape[right_s:right_e]
                left_eye_EAR = self.eye_aspect_ratio(leftEye)
                right_eye_EAR = self.eye_aspect_ratio(rightEye)
                ear = (left_eye_EAR + right_eye_EAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                count = 0
                # TODO va da 0.10 a 0.29
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

                # if ear > ear_top:
                #     ear_top = ear

            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
            # TODO servono counter e total?
            return eyes_detect, COUNTER, TOTAL, ear_th


def main():
    EyeBlink(None).eyeBlinkStart()


if __name__ == '__main__':
    main()
