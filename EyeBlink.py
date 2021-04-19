#######################################################
#Qui abbiamo la classe EyeBlink il cui compito è quello
#di rilevare se uno utente tramite video o tramite
#webcam sta battendo le palpebre oppure no.
#######################################################

import cv2
import imutils
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from scipy.spatial import distance as dist
from imutils.video import FPS
import time
import numpy as np
from scipy.spatial import distance

from imutils import face_utils
import dlib



EYE_AR_THRESH = 0.28 #Thresh iniziale
CONSEC_FRAMES_NUMBER = 3



class EyeBlink():
    def __init__(self, inputType):
        self.inputType = inputType



    #Il metodo sottostante va a prendere un frame per volta dal video
    # (se è stato passato un input alla classe) oppure utilizza come
    # sorgente la webcam e passa i frame ai relativi metodi "eye_blink_video"(se si tratta di un video)
    # o "eye_blink_cam" se la sorgente è una webcam tali metodi andranno ad analizzare i singoli frame e ritornano
    # dei parametri come l'EAR_TOP ( che è Eye_aspect_ratio più alto che si incontra durante l'analisi dei frame
    # di uno streaming video), il conteggio di blinking e l'eventuale history se abbiamo ottenuto almeno un eyeblink
    # ritorniamo True altrimenti continuiamo fino alla fine del video e se non è stato mai rilevato un eyeblinking
    # torna False

    def eyeBlinkStart(self):
        inputType = self.inputType
        COUNTER = 0
        TOTAL = 0
        if inputType is not None:


            ear_top = 0


            history = ' '
            print(inputType)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

            fvs = FileVideoStream(inputType).start()
            time.sleep(1.0)
            fps = FPS().start()



            fileStream = True

            while fvs.more():
                crop = None
                frame = fvs.read()

                try:
                    frame = imutils.resize(frame, width=300)

                except Exception as e:
                    print(str(e))
                    if (len(history) > 1):
                        print(history)
                        result = self.isBlinking(history, 3)
                        if (result):
                            cv2.destroyAllWindows()
                            fvs.stop()
                            return True
                        else:
                            cv2.destroyAllWindows()
                            fvs.stop()
                            return False
                try:
                    vis = frame.copy()

                except Exception as e:
                    print(str(e))
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                frame = np.dstack([frame, frame, frame])

                eyesdetect, COUNTER, TOTAL, ear_top = self.eye_blink_video(frame, detector, predictor,
                                            COUNTER, TOTAL, ear_top)
                history += eyesdetect
                if TOTAL > 0:
                    var = True
                    cv2.destroyAllWindows()
                    fvs.stop()
                    return var
                else:
                    var = False
                fps.update()
                # print(len(history))
                if (len(history) > 250):

                    print(history)
                    result = self.isBlinking(history, 3)
                    print(result)
                    if (result and var == False):
                        cv2.destroyAllWindows()
                        fvs.stop()
                        # cv2.putText(vis, "Real", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                        return True
                    else:
                        cv2.destroyAllWindows()
                        fvs.stop()
                        # cv2.putText(vis, "Fake", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                        return False


                fps.stop()
                print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
                print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

            cv2.destroyAllWindows()
            fvs.stop()

        elif inputType is None:
            cap = cv2.VideoCapture(0)
            # Write the label with this font
            history = ''
            font = cv2.FONT_HERSHEY_SIMPLEX

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            ear_top = 0
            while (True):

                ret, frame = cap.read()

                vis = frame.copy()


                eyedetect, TOTAL, COUNTER, ear_top = self.eye_blink_cam(frame, ret, detector, predictor,
                                                COUNTER, TOTAL,ear_top)
                history += eyedetect
                if TOTAL >= 1:
                  cv2.putText(frame, "Real", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                elif (len(history) > 100):
                        print(history)
                        result = self.isBlinking(history, 3)
                        print(result)
                        if (result):
                            pass
                            #superfluo
                            #cv2.putText(frame, "Real", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                        else:
                            cv2.putText(frame, "Fake", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                        #cv2.imshow('Face', vis)
                else:
                  cv2.putText(frame, "Check...", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 3, cv2.LINE_AA)

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

    #ToDo non so se lasciare questa funzione superflua
    def isBlinking(self, history, maxFrames):
        for i in range(maxFrames):
            pattern = '1' + '0' * (i + 1) + '1'
            if pattern in history:
                return True
        return False

    def eye_blink_cam(self, frame, rect, detector, predictor, COUNTER, TOTAL, ear_top):


        eyes_detect = ''
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)  # Detect the faces in the image
        (left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



        for det in dets:
            shape = predictor(gray, det)
            shape = face_utils.shape_to_np(shape)
            ###
            (x, y, w, h) = face_utils.rect_to_bb(det)
            cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 255, 0), 2)  # vis-->rect
            cv2.putText(rect, "Face", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # vis-->rect
            ###
            leftEye = shape[left_s:left_e]
            rightEye = shape[right_s:right_e]
            left_eye_EAR = self.eye_aspect_ratio(leftEye)
            right_eye_EAR = self.eye_aspect_ratio(rightEye)
            ear = (left_eye_EAR + right_eye_EAR) / 2.0


            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            ear_threshold = (ear_top * 2)/3
            print("Ear_th", ear_threshold)
            print("EAR TOP", ear_top)
            for (x, y) in shape:
                cv2.circle(rect, (x, y), 1, (0, 0, 255), -1)

            if ear_top != 0:
                ear_threshold = (ear_top * 2) / 3
                print("Ear_th", ear_threshold)
                print("EAR TOP", ear_top)

                if ear < ear_threshold:

                    eyes_detect = '1'
                    COUNTER +=1
                else:
                    eyes_detect = '0'

                    if COUNTER >= ear_threshold:
                        TOTAL += 1



                    COUNTER = 0



            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if ear > ear_top:
                ear_top = ear

        #cv2.imshow('Frame', frame)
        #cv2.waitKey(1)
        return eyes_detect, TOTAL, COUNTER, ear_top

    def eye_blink_video(self, frame, detector, predictor, COUNTER, TOTAL, ear_top):

        eyes_detect = ''

        rects = detector(frame, 1)  # Detect the faces in the image
        (left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



        for rect in rects:
            shape = predictor(frame, rect)
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


                if ear < ear_threshold:
                    COUNTER += 1
                    print(COUNTER)
                    eyes_detect = '1'
                else:
                    eyes_detect = '0'
                    if COUNTER >= ear_threshold:

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


def main():
    EyeBlink(None).eyeBlinkStart()

if __name__ == '__main__':
    main()


