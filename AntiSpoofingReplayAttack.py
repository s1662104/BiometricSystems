import cv2
import imutils

import AntiSpoofingSplitting
import Main
import LBP
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from mlxtend.plotting import plot_decision_regions


def train_svm(X_train,y_train,X_test,y_test):
    model = SVC(kernel='rbf', random_state=0,gamma=1,C=1 )
    svm = model.fit(X_train, y_train)


    # Evaluate by means of a confusion matrix
    matrix = plot_confusion_matrix(svm, X_test, y_test,
                                   cmap=plt.cm.Blues,
                                   normalize='true')
    plt.title('Confusion matrix for RBF SVM')
    plt.show(matrix)
    plt.show()

    # Generate predictions
    y_pred = svm.predict(X_test)

    # Evaluate by means of accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy}')

    # Plot decision boundary
    plot_decision_regions(X_test, y_test, clf=svm, legend=2)
    plt.show()






##############################################
#Main locale che dovrà essere poi sostituito
############################################
def main():
    cap = cv2.VideoCapture(0)
    # Write the label with this font
    split_width = 22
    split_height = 22

    img_h = 0
    img_w = 0
    while (True):
        crop = None
        ret,frame = cap.read()


        vis = frame.copy()


        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = imutils.resize(frame, width=150)
        crop = Main.detect_face(gray,vis)

        if crop is not None:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detector = dlib.get_frontal_face_detector()
        # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            myLBP = LBP.Spoof_Local_Binary_Pattern(1, 8, crop)


        new_img = myLBP.compute_lbp()
        hist = myLBP.createHistogram(new_img)
        svm = AntiSpoofingSplitting.main()
        nsamples = hist.shape
        print("nsamples",nsamples)
        hist = hist.reshape(1,-1)
        print("histogram")
        print(hist)
        value = (svm.predict(hist))
        print(value)
        if value == 0:
            print("REAL")
        else:
            print("FAKE")
        return

        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.show()






        #new code (rect)

        #end code

        # history += eye_blink(frame,ret)
        #
        # if(len(history)> 10):
        #     print(history)
        #     result = isBlinking(history,3)
        #     print(result)
        #     if(result):
        #         cv2.putText(vis, "Real", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        #     else:
        #         cv2.putText(vis, "Fake", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #     cv2.imshow('Face', vis)
        #
        # #cv2.imshow("Frame", frame)
        # cv2.imshow("Face", vis)
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break

        # if the `q` key was pressed, break from the loop
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()