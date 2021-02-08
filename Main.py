import dlib
import numpy as np
import cv2

message = "Inserire codice fiscale: "
error = "Codice fiscale non valido"
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
        crop = cv2.resize(crop, (dim_image, dim_image))
    if (len(dets)>0):
        cv2.imshow('Face', crop)



if __name__ == '__main__':
    cf = input(message)
    if (len(cf)!=16): print(error)
    else:
        cap = cv2.VideoCapture(0)

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            vis = frame.copy()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detect_face(gray, vis)

            # Display the resulting frame
            cv2.imshow('frame', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()