import cv2
import imutils
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from scipy.spatial import distance as dist
from imutils.video import FPS
import time
import Main
import numpy as np
from scipy.spatial import distance
#import matplotlib.pyplot as plt
from imutils import face_utils
import dlib
EYE_AR_THRESH = 0.3
CONSEC_FRAMES_NUMBER = 3

class EyeBlink():
    def __init__(self, inputType):
        self.inputType = inputType
        # Function to calculate EAR
    def eyeBlinkStart(self):
        inputType = self.inputType
        if inputType is not None:
            history = ' '
            print(inputType)
            fvs = FileVideoStream(inputType).start()
            fps = FPS().start()


            #cap = cv2.VideoCapture(inputType)
            #fileStream = True
            time.sleep(1.0)
            # if(cap.isOpened() == False):
            #     print("Error opening video stream or file")
            while(fvs.more):
                # if fileStream and not vs.more():
                #     break

                frame = fvs.read()
                # ret , frame = cap.read()
                if frame is None:
                    print("frame is None")

                frame = fvs.read()
                frame = imutils.resize(frame, width=450)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.dstack([frame, frame, frame])


                # try:
                #     vis = frame.copy
                # except Exception as e:
                #     print(str(e))
                #     break

                history += self.eye_blink_video(frame)
                fps.update()
                if (len(history) > 100):
                    print(history)
                    result = self.isBlinking(history, 3)
                    print(result)
                    if (result):
                        #cv2.putText(vis, "Real", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                        return True
                    else:
                        #cv2.putText(vis, "Fake", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                        return False
                    #cv2.imshow('Face', vis)

                # cv2.imshow("Frame", frame)
                #cv2.imshow("Face", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                fps.stop()
                print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
                print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
                # if the `q` key was pressed, break from the loop
            # cap.release()
            cv2.destroyAllWindows()
            fvs.stop()

        elif inputType is None:
            cap = cv2.VideoCapture(0)
            # Write the label with this font
            history = ''
            font = cv2.FONT_HERSHEY_SIMPLEX
            COUNT = 0
            TOTAL = 0

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

            while (True):

                ret, frame = cap.read()

                vis = frame.copy()

                # frame = imutils.resize(frame, width=640)
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # detector = dlib.get_frontal_face_detector()
                # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

                # new code (rect)

                # end code

                history += self.eye_blink_cam(frame, ret, detector, predictor)


                if (len(history) > 10):
                    print(history)
                    result = self.isBlinking(history, 3)
                    print(result)
                    if (result):
                        cv2.putText(vis, "Real", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(vis, "Fake", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.imshow('Face', vis)

                # cv2.imshow("Frame", frame)
                cv2.imshow("Face", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # if the `q` key was pressed, break from the loop
            cap.release()
            cv2.destroyAllWindows()
        else:
            exit("Il valore di inputType è errato")

    #EAR = (||p2 - p6||+ ||p3 - p5||)/(2||p1-p4||)
    def eye_aspect_ratio(self,eye):
        A = distance.euclidean(eye[1],eye[5])
        B = distance.euclidean(eye[2],eye[4])
        C = distance.euclidean(eye[0],eye[3])
        EAR = (A + B) / (2.0 * C)
        return EAR

    def isBlinking(self,history, maxFrames):
        for i in range(maxFrames):
            pattern = '1' + '0'*(i+1) + '1'
            if pattern in history:
                return True
        return False

    def eye_blink_cam(self,frame,rect, detector, predictor):
        #frame = imutils.resize(frame ,width = 640)
        crop = None
        eyes_detect = ''
        frame = imutils.resize(frame, width = 640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)  # Detect the faces in the image
        (left_s,left_e) =face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        ear = 0
        #history = {}

        for det in dets:
            shape = predictor(gray,det)
            shape = face_utils.shape_to_np(shape)
            ###
            (x,y,w,h) = face_utils.rect_to_bb(det)
            cv2.rectangle(rect, (x,y), (x+w, y+h), (0, 255, 0), 2) #vis-->rect
            cv2.putText(rect, "Face",(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 255, 0), 2) #vis-->rect
            ###
            leftEye = shape[left_s:left_e]
            rightEye = shape[right_s:right_e]
            left_eye_EAR = self.eye_aspect_ratio(leftEye)
            right_eye_EAR = self.eye_aspect_ratio(rightEye)
            ear = (left_eye_EAR+right_eye_EAR)/2.0

            #draw the faceDetect
            # top = shape[19]
            # left = shape.part(0).x
            # right = shape.part(16).x
            # bottom = shape.part(8).y
            # crop = frame[top:bottom, left:right]
            #
            # crop = cv2.resize(crop, (64, 64))
            # visualize each of the eyes
            # leftEyeHull = cv2.convexHull(leftEye)
            # rightEyeHull = cv2.convexHull(rightEye)
            # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            for (x,y) in shape:
                cv2.circle(rect, (x,y), 1, (0,0,255), -1)

            if ear < EYE_AR_THRESH:
                eyes_detect ='1'
            else:
                eyes_detect ='0'

            #        COUNTER = COUNTER + 1
            # else:
            #     if COUNTER >= CONSEC_FRAMES_NUMBER:
            #         TOTAL = TOTAL + 1
            #     COUNTER = 0

        # cv2.putText(gray, "Blinks: {}".format(TOTAL), (10, 30),
        #             font, 0.7, (0, 255, 0), 2)
        # cv2.putText(gray, "EAR: {:.2f}".format(ear), (500, 30),
        #             font, 0.7, (0, 255, 0), 2)

        cv2.imshow('Face', rect)
        return eyes_detect

    def eye_blink_video(self,frame):
        crop = None
        eyes_detect = ''
        #print(frame)
        # frame = imutils.resize(frame, width=450)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        rects = detector(frame, 0)  # Detect the faces in the image
        (left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        ear = 0
        # history = {}
        ###
        # cv2.putText(frame, "Slow Method", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # show the frame and update the FPS counter
        # cv2.imshow("Frame", frame)
        # cv2.waitKey(1)


        for rect in rects:
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)
            ###
            # (x, y, w, h) = face_utils.rect_to_bb(det)
            # cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 255, 0), 2)  # vis-->rect
            # cv2.putText(rect, "Face", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # vis-->rect
            ###
            leftEye = shape[left_s:left_e]
            rightEye = shape[right_s:right_e]
            left_eye_EAR = self.eye_aspect_ratio(leftEye)
            right_eye_EAR = self.eye_aspect_ratio(rightEye)
            ear = (left_eye_EAR + right_eye_EAR) / 2.0

            # draw the faceDetect
            # top = shape[19]
            # left = shape.part(0).x
            # right = shape.part(16).x
            # bottom = shape.part(8).y
            # crop = frame[top:bottom, left:right]
            #
            # crop = cv2.resize(crop, (64, 64))
            # visualize each of the eyes
            # leftEyeHull = cv2.convexHull(leftEye)
            # rightEyeHull = cv2.convexHull(rightEye)
            # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            # for (x, y) in shape:
            #     cv2.circle(rect, (x, y), 1, (0, 0, 255), -1)

            if ear < EYE_AR_THRESH:
                eyes_detect = '1'
            else:
                eyes_detect = '0'

            #        COUNTER = COUNTER + 1
            # else:
            #     if COUNTER >= CONSEC_FRAMES_NUMBER:
            #         TOTAL = TOTAL + 1
            #     COUNTER = 0

        # cv2.putText(gray, "Blinks: {}".format(TOTAL), (10, 30),
        #             font, 0.7, (0, 255, 0), 2)
        # cv2.putText(gray, "EAR: {:.2f}".format(ear), (500, 30),
        #             font, 0.7, (0, 255, 0), 2)

        #cv2.imshow('Face', frame)
        return eyes_detect

#LA PRIMA COSA DA FARE E' RILEVARE LA FACCIA NELL'IMMAGINE

##########################################
###Questa parte qui sotto dovrà essere modificata e fatta partire dal main###
##########################################


