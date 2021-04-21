import cv2
import dlib
import pandas as pd
import numpy as np
import LBP
import math
import random
import ast

#MZZLSN95R16H501Q

threshold = 0.6

#face identification (or 1:N face recognition) consists in finding the identity corresponding to a given face
def identify(cf, img):

    #upload the various datasets
    gallery_data = np.load("npy_db/gallery_data.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    users = pd.read_csv('dataset_user.csv', index_col=[0])

    #find the user linked to the cf
    cf_list = users['Codice Fiscale']
    index = cf_list.tolist().index(cf)
    user = users.iloc[index]
    print(user)

    #find the user's delegates
    delegati = ast.literal_eval(user["Delegati"])
    print(delegati)

    #we begin to find out which delegate is trying to access to the system
    max = 0
    identity = None
    norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    for d in delegati:
        print("Delegato",d)
        #the best value obtained by comparing the input image with the delegate images in the gallery
        val = topMatch(norm_image,d,gallery_data,gallery_target)
        if val > max:
            max = val
            identity = d    #the identity of the delegate who gets the best value for the moment
    print("L'identità del delegato è:",identity)
    return user, index


#face verification (or 1:1 face recognition) consists in checking if a face corresponds to a given identity
def recognize(cf, img):

    #upload the various datasets
    gallery_data = np.load("npy_db/gallery_data.npy")
    gallery_target  = np.load("npy_db/gallery_target.npy")
    users = pd.read_csv('dataset_user.csv', index_col=[0])

    #find the user linked to the cf
    cf_list = users['Codice Fiscale']
    user = None
    index = 0

    #check if the user is registered
    if cf_list.tolist().__contains__(cf):
        print("UTENTE PRESENTE")
    else:
        print("UTENTE NON PRESENTE! Il codice fiscale inserito è:",cf)
        return

    #normalize the input image
    norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)

    #calculates the maximum match between the image taken in input and the images in the user's gallery
    val = topMatch(norm_image,cf,gallery_data,gallery_target)

    #if the maximum match is greater than the threshold, then the identity is verified
    if val > 0.6:
        index = cf_list.tolist().index(cf)
        user = users.iloc[index]

    return user, index

def testRecognition(user, template):

    # upload the various datasets
    gallery_data = np.load("npy_db/gallery_data.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")

    #compute LBP to calculate the histogram of the probe image
    lbp_probe = LBP.Local_Binary_Pattern(1, 8, template)
    new_img = lbp_probe.compute_lbp()
    hist_probe = lbp_probe.createHistogram(new_img)
    compareHistogram(hist_probe, hist_probe)

    print("Verifichiamo l'identità di",user)

    #check if the user is registered
    if gallery_target.tolist().__contains__(user):
        n = gallery_target.tolist().count(user)     #num dei template dell'utente presenti nella gallery
        print("Numero di template presenti nella gallery:",n)
        i = gallery_target.tolist().index(user)     #index of the user's first template in the gallery
        print("Primo indice:", i)
        max = 0
        for j in range(0, n):
            #compute LBP to calculate the histogram of the gallery image
            lbp_gallery = LBP.Local_Binary_Pattern(1, 8, gallery_data[i+j])
            list_hist_user = lbp_probe.createHistogram(lbp_gallery.compute_lbp())
            diff = compareHistogram(hist_probe,list_hist_user)
            if diff >= max: max = diff
            print("\n")

        print("Il valore maggiore ottenuto tra gli istogrammi:", max)

    #if the maximum matching is greater than the threshold, then the identity is verified
    if max > threshold: print("L'IDENTITA' DELL'UTENTE E' STATA VERIFICTA CORRETTAMENTE")
    else: print("L'IDENTITA' DICHIARATA NON CORRIISPONDE")
    return

def compareHistogram(H1, H2):

    #in order to make the difference between the two histograms, we must be sure that their length is the same
    if len(H1) != len(H2):
        print("LA LUNGHEZZA DEI DUE ISTOGRAMMI NON E' LA STESSA")
        return

    #calculate the average of the values in the two histograms
    avg_H1 = sum(H1) / len(H1)
    avg_H2 = sum(H2) / len(H2)

    sum1 = 0
    sum_H1 = 0
    sum_H2 = 0
    for i in range(0, len(H1)):
        sum1 = sum1 + ((H1[i]-avg_H1)*(H2[i]-avg_H2))
        sum_H1 = sum_H1 + pow(H1[i]-avg_H1, 2)
        sum_H2 = sum_H2 + pow(H2[i]-avg_H2, 2)

    #calculate the difference using the correlation method
    d = sum1 / math.sqrt(sum_H1*sum_H2)
    print("La differenza tra i due istogrammi ottenuta usando la Correlation è:",d)

    return d

#False Rejection Rate - FRR
#The FRR is defined as the percentage of identification instances in which false rejection occurs.
#This can be expressed as a probability. For example, if the FRR is 0.05 percent, it means that on the average,
#one out of every 2000 authorized persons attempting to access the system will not be recognized by that system.
def verificationFRR():
    pg_target = np.load("npy_db/pg_target.npy")
    pg_data = np.load("npy_db/pg_data.npy")
    gallery_data = np.load("npy_db/gallery_data.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    P = 0
    for i in range(len(pg_data)):
        pg_template = pg_data[i]
        pg_identity = pg_target[i]
        #topMatch(p, identity) returns the best match between pj and the templates associated to the claimed identity in the gallery
        gx = topMatch(pg_template, pg_identity, gallery_data, gallery_target)
        if gx <= threshold:
            print("L'identità dichiarata",pg_identity,"non è stata accettata")
            print("Il valore di similarity ottenuto è:", gx)
            P = P + 1
        else:
            print("L'dentità dichiarata", pg_identity, "è stata accettata")
            print("Il valore di similarity ottenuto è:", gx)
    print("Il numero di identità rifiutate è:",P)
    print("FRR:", P/len(pg_data))
    return

#False Acceptance Rate - FAR
#The FAR is defined as the percentage of identification instances in which false acceptance occurs.
#This can be expressed as a probability. For example, if the FAR is 0.1 percent, it means that on the average, one out of every 1000
#impostors attempting to breach the system will be successful. Stated another way, it means that the probability of an unauthorized person being identified as
#an authorized person is 0.1 percent.
def verificationFAR():
    pg_target = np.load("npy_db/pg_target.npy")
    pn_target = np.load("npy_db/pn_target.npy")
    pn_data = np.load("npy_db/pn_data.npy")
    gallery_data = np.load("npy_db/gallery_data.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    P = 0
    #Scenario in which the impostor doesn't belong to the gallery
    for i in range(len(pn_data)):
        pn_template = pn_data[i]
        pn_identity = pn_target[i]
        pg_identity = pg_target[random.randrange(len(pg_target))]
        #topMatch(p, identity) returns the best match between pj and the templates associated to the claimed identity in the gallery
        gx = topMatch(pn_template, pg_identity, gallery_data, gallery_target)
        if gx > threshold:
            print("L'identità dichiarata", pg_identity, "è stata accettata")
            print("La vera identità però è:",pn_identity)
            print("Il valore di similarity ottenuto è:", gx)
            P = P + 1
        else:
            print("L'dentità dichiarata", pg_identity, "non è stata accettata")
            print("La vera identità però è:", pn_identity)
            print("Il valore di similarity ottenuto è:", gx)
    print("Il numero di identità errate accettate è:", P)
    print("FAR:", P / len(pn_data))

    return

def topMatch(probe, identity, gallery_data, gallery_target):
    max = 0
    lbp_probe = LBP.Local_Binary_Pattern(1, 8, probe)
    new_img = lbp_probe.compute_lbp()
    hist_probe = lbp_probe.createHistogram(new_img)
    for i in range(len(gallery_data)):
        if gallery_target[i] == identity:
            lbp_gallery = LBP.Local_Binary_Pattern(1, 8, gallery_data[i])
            hist_gallley = lbp_probe.createHistogram(lbp_gallery.compute_lbp())
            diff = compareHistogram(hist_probe, hist_gallley)
            if diff >= max:
                max = diff
    return max

def main():
    pg_target = np.load("npy_db/pg_target.npy")
    pg_data = np.load("npy_db/pg_data.npy")
    pn_data = np.load("npy_db/pn_data.npy")

    print("Il template utilizzato è quello dell'utente, quindi la verificazione dovrebbe risultare positiva")
    testRecognition(pg_target[0], pg_data[0])
    print("Il template utilizzato non è quello dell'utente, quindi la verificazione dovrebbe risultare negativa")
    testRecognition(pg_target[0], pn_data[0])

if __name__ == '__main__':
    #main()
    #verificationFRR()
    #verificationFAR()
    cf = "NTHKRN91B30G259C"
    crop = None
    video_capture = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    while True:
        ret, frame = video_capture.read()
        dets = detector(frame, 1)
        for i, d in enumerate(dets):
            landmark = predictor(frame, d)
            top = landmark.part(19).y
            left = landmark.part(0).x
            right = landmark.part(16).x
            bottom = landmark.part(8).y
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            crop = frame[top:bottom, left:right]
            crop = cv2.resize(crop, (64, 64))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            break
    video_capture.release()
    cv2.destroyAllWindows()
    identify(cf, crop)
