import dlib
import numpy as np
import cv2

message = "Inserire codice fiscale: "
error = "Codice fiscale non valido"

def detect_face(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    dets = detector(img, 1)  # Detect the faces in the image
    crop=img
    for i, d in enumerate(dets):
        landmark = predictor(img, d)
        dist_eye = landmark.part(45).x- landmark.part(36).x
        print(dist_eye)
        crop = img[d.top():d.bottom(), d.left():d.right()]
        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 3)
    cv2.imshow('Face', crop)



if __name__ == '__main__':
    cf = input(message)
    if (len(cf)!=16): print(error)
    else:
        cap = cv2.VideoCapture(0)

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detect_face(gray)

            # Display the resulting frame
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()