import cv2
import pandas as pd
import numpy as np
import LBP
import math
import ast
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

threshold = 0.55

#face identification (or 1:N face recognition) consists in finding the identity corresponding to a given face.
def identify(cf, img):

    #upload the various datasets
    gallery_data = np.load("npy_db/gallery_data.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    users = pd.read_csv('dataset_user.csv', index_col=[0])

    #find the user linked to the cf
    cf_list = users['Codice Fiscale']
    index = cf_list.tolist().index(cf)
    user = users.iloc[index]
    print(user)
    print(user["Delegati"])
    #find the user's delegates
    delegati = ast.literal_eval(user["Delegati"])
    print(delegati)

    #if the user doesn't have delegates
    if len(delegati) == 0:
        print("L'utente non ha delegati!")
        return None, 0

    #we begin to find out which delegate is trying to access to the system
    max = 0
    identity = None
    norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    lbp = LBP.Local_Binary_Pattern(1, 8, norm_image)
    hist = lbp.createHistogram(lbp.compute_lbp())
    for d in delegati:
        #the best value obtained by comparing the input image with the delegate images in the gallery
        val = topMatch(d, gallery_target, histogram_gallery_data, hist)
        if val > max  and val > threshold:
            max = val
            identity = d    #the identity of the delegate who gets the best value for the moment
    print("L'identità del delegato è:",identity)
    if identity is not None:
        indexd = cf_list.tolist().index(identity)
        recUser = users.iloc[indexd]
    return user, index, recUser

#face verification (or 1:1 face recognition) consists in checking if a face corresponds to a given identity.
def recognize(cf, img):

    #upload the various datasets
    gallery_data = np.load("npy_db/gallery_data.npy")
    gallery_target  = np.load("npy_db/gallery_target.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
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
    lbp = LBP.Local_Binary_Pattern(1, 8, norm_image)
    hist = lbp.createHistogram(lbp.compute_lbp())

    #calculates the maximum match between the image taken in input and the images in the user's gallery
    val = topMatch(cf, gallery_target, histogram_gallery_data, hist)
    print("val:",val)
    #if the maximum match is greater than the threshold, then the identity is verified
    if val > threshold:
        index = cf_list.tolist().index(cf)
        user = users.iloc[index]

    return user, index, user

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
    #print("La differenza tra i due istogrammi ottenuta usando la Correlation è:",d)

    return d

#False Rejection Rate - FRR
#The FRR is defined as the percentage of identification instances in which false rejection occurs.
#This can be expressed as a probability. For example, if the FRR is 0.05 percent, it means that on the average,
#one out of every 2000 authorized persons attempting to access the system will not be recognized by that system.
def verificationFRR():
    pg_target = np.load("npy_db/pg_target.npy")
    pg_data = np.load("npy_db/pg_data.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data =  np.load("npy_db/histogram_pg_data.npy")
    P = 0
    for i in range(len(pg_data)):
        pg_identity = pg_target[i]
        pg_hist = histogram_pg_data[i]
        #topMatch(p, identity) returns the best match between pj and the templates associated to the claimed identity in the gallery
        gx = topMatch(pg_identity, gallery_target, histogram_gallery_data, pg_hist)
        if gx < gallery_thresholds[int(i/5)]:
            P = P + 1
    #print("Il numero di identità rifiutate è:",P)
    print("FRR:", P/len(pg_data))
    return

#False Acceptance Rate - FAR
#The FAR is defined as the percentage of identification instances in which false acceptance occurs.
#This can be expressed as a probability. For example, if the FAR is 0.1 percent, it means that on the average, one out of every 1000
#impostors attempting to breach the system will be successful. Stated another way, it means that the probability of an unauthorized person being identified as
#an authorized person is 0.1 percent.
def verificationFAR():
    pg_target = np.load("npy_db/pg_target.npy")
    pn_data = np.load("npy_db/pn_data.npy")
    pg_data = np.load("npy_db/pn_data.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data = np.load("npy_db/histogram_pg_data.npy")
    histogram_pn_data = np.load("npy_db/histogram_pn_data.npy")
    P = 0
    ti = 0
    #Scenario in which the impostor doesn't belong to the gallery
    for i in range(len(pn_data)):
        pn_hist = histogram_pn_data[i]
        index_target = 0
        for t in np.unique(gallery_target):
            #topMatch(p, identity) returns the best match between pj and the templates associated to the claimed identity in the gallery
            val = topMatch(t, gallery_target, histogram_gallery_data, pn_hist)
            if val >= gallery_thresholds[index_target]:
                P = P + 1
            ti += 1
            index_target += 1

    #Scenario in which the impostor belongs to the gallery
    for i in range(len(pg_data)):
        pg_hist = histogram_pg_data[i]
        for t in np.unique(gallery_target):
            index_target = 0
            if t != pg_target[i]:
                # topMatch(p, identity) returns the best match between pj and the templates associated to the claimed identity in the gallery
                val = topMatch(t, gallery_target, histogram_gallery_data, pg_hist)
                if val >= gallery_thresholds[index_target]:
                    P = P + 1
                ti += 1
            index_target += 1

    #print("Il numero di identità errate accettate è:", P)
    #print("FAR:", P / (len(pn_data)*len(np.unique(gallery_target))))
    print("FAR:", P / ti)

    return

#The Receiver Operating Characteristics (ROC) curve is an evaluation metric for a binary classifier,
#which helps us to visualize the performance of a facial recognition model as its discrimination threshold changes.
#ROC depicts the probability of Genuine Accept (GAR) of the system, expressed as 1-FRR, vs False Accept Rate (FAR) variation.
def verificationROC():
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    pg_data = np.load("npy_db/pg_data.npy")
    pg_target = np.load("npy_db/pg_target.npy")
    pn_data = np.load("npy_db/pn_data.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data = np.load("npy_db/histogram_pg_data.npy")
    histogram_pn_data = np.load("npy_db/histogram_pn_data.npy")

    Y = []
    Y_pred = []

    for i in range(len(pg_data)):
        Y.append(1)
        val = topMatch(pg_target[i], gallery_target, histogram_gallery_data, histogram_pg_data[i])
        if val >= gallery_thresholds[int(i/5)]:
            Y_pred.append(1)
        else:
            Y_pred.append(0)

    for j in range(len(pn_data)):
        index_target = 0
        for t in np.unique(gallery_target):
            Y.append(0)
            val = topMatch(t, gallery_target, histogram_gallery_data, histogram_pn_data[j])
            if val >= gallery_thresholds[index_target]:
                Y_pred.append(1)
            else:
                Y_pred.append(0)
            index_target += 1

    for v in range(len(pg_data)):
        index_target = 0
        for u in np.unique(gallery_target):
            if pg_target[v] != u:
                Y.append(0)
                val = topMatch(u, gallery_target, histogram_gallery_data, histogram_pg_data[v])
                if val >= gallery_thresholds[index_target]:
                    Y_pred.append(1)
                else:
                    Y_pred.append(0)
            index_target += 1

    FPR, TPR, t = roc_curve(np.array(Y), np.array(Y_pred))
    auc = roc_auc_score(np.array(Y), np.array(Y_pred))

    FNR = 1 - TPR
    EER = FPR[np.nanargmin(np.absolute((FNR - FPR)))]
    print("EER:",EER)

    #Plot ROC curve
    plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return

#MODIFICA IN MODO DA TOGLIERE IL FOR
def topMatch(identity , gallery_target, histogram_gallery_data, hist):
    max = 0
    #lbp_probe = LBP.Local_Binary_Pattern(1, 8, probe)
    #new_img = lbp_probe.compute_lbp()
    #hist_probe = lbp_probe.createHistogram(new_img)
    index = gallery_target.tolist().index(identity)
    for i in range(5):
        #lbp_gallery = LBP.Local_Binary_Pattern(1, 8, gallery_data[index+i])
        #hist_gallley = lbp_probe.createHistogram(lbp_gallery.compute_lbp())
        #diff = compareHistogram(hist_probe, hist_gallley)
        #diff = compareHistogram(hist_probe, histogram_gallery_data[index+i])
        diff = compareHistogram(hist, histogram_gallery_data[index + i])
        if diff >= max:
            max = diff

    # for i in range(len(gallery_data)):
    #     if gallery_target[i] == identity:
    #         lbp_gallery = LBP.Local_Binary_Pattern(1, 8, gallery_data[i])
    #         hist_gallley = lbp_probe.createHistogram(lbp_gallery.compute_lbp())
    #         diff = compareHistogram(hist_probe, hist_gallley)
    #         if diff >= max:
    #             max = diff

    if max == 0:
        print("L'utente",identity,"non è nella gallery")
    return max

# def evaluationIdentification():
#     gallery_data = np.load("npy_db/gallery_data.npy")
#     gallery_target = np.load("npy_db/gallery_target.npy")
#     pg_data = np.load("npy_db/pg_data.npy")
#     pg_target = np.load("npy_db/pg_target.npy")
#     pn_data = np.load("npy_db/pn_data.npy")
#     pn_target = np.load("npy_db/pn_target.npy")
#     users = pd.read_csv('dataset_user.csv', index_col=[0])
#     cf_list = users['Codice Fiscale']
#     count = 0
#     di = [0]*len(np.unique(gallery_target))
#     fa = 0
#     gr = 0
#     for i in range(len(pg_data)):
#         probe_target = pg_target[i]
#         probe_template = pg_data[i]
#         list = []
#         for j in range(len(np.unique(gallery_target))):
#             cf_user = np.unique(gallery_target)[j]
#             index = cf_list.tolist().index(cf_user)
#             user = users.iloc[index]
#             delegati = ast.literal_eval(user["Delegati"])
#             #if the user has delegates
#             if len(delegati) > 0:
#                 count += 1
#                 max = 0
#                 for t in delegati:
#                     val = topMatch(probe_template, t, gallery_data,gallery_target)
#                     if val > max:
#                         max = val
#                 list.append((cf_user, max, delegati))
#         list = sorted(list, key=lambda x: x[1], reverse=True)
#         print(list)
#         if list[0][1] >= threshold:
#             if probe_target in list[0][2]:
#                 di[0] += 1
#             else:
#                 for v in range(1,len(list)):
#                     if probe_target in list[v][2] and list[v][1] >= threshold:
#                         di[v] += 1
#                 fa += 1
#         else:
#             gr += 1
#
#     dir = [0]*len(np.unique(gallery_target))
#     dir[0] = di[0]/count
#     frr = 1 - dir[0]
#     print("DIR:",dir)
#     print("FRR:",frr)
#
#     return

def evaluationIdentificationAsMultiVer():
    gallery_data = np.load("npy_db/gallery_data.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data = np.load("npy_db/histogram_pg_data.npy")
    histogram_pn_data = np.load("npy_db/histogram_pn_data.npy")
    pg_data = np.load("npy_db/pg_data.npy")
    pg_target = np.load("npy_db/pg_target.npy")
    pn_data = np.load("npy_db/pn_data.npy")
    pn_target = np.load("npy_db/pn_target.npy")
    users = pd.read_csv('dataset_user.csv', index_col=[0])
    cf_list = users['Codice Fiscale']
    #results1 = delegatesMatch(pg_data,pg_target,gallery_target,gallery_data,cf_list,users, gallery_thresholds, histogram_gallery_data)
    #results2 = delegatesMatch(pn_data, pn_target, gallery_target, gallery_data, cf_list, users, gallery_thresholds, histogram_gallery_data)
    results1 = delegatesMatch(histogram_pg_data,pg_target,gallery_target,gallery_data,cf_list,users, gallery_thresholds, histogram_gallery_data)
    results2 = delegatesMatch(histogram_pn_data, pn_target, gallery_target, gallery_data, cf_list, users, gallery_thresholds, histogram_gallery_data)

    print("PG_DATA:",results1)
    print("PN_DATA:", results2)

    fa = results1[0] + results2[0]
    fr = results1[1] + results2[1]
    countTG = results1[2] + results2[2]
    countTI = results1[3] + results2[3]
    FRR = fr/countTG
    FAR = fa/countTI
    print("FRR:", FRR, countTG)
    print("FAR:", FAR, countTI)

#def delegatesMatch(data, target, gallery_target, gallery_data, cf_list, users, gallery_thresholds, histogram_gallery_data):
def delegatesMatch(hist_data, target, gallery_target, gallery_data, cf_list, users, gallery_thresholds, histogram_gallery_data):
    countTG = 0
    countTI = 0
    fa = 0
    fr = 0
    #for i in range(len(data)):
    for i in range(len(hist_data)):
        probe_target = target[i]
        #probe_template = data[i]
        hist_probe = hist_data[i]
        for j in range(len(np.unique(gallery_target))):
                cf_user = np.unique(gallery_target)[j]
                index = cf_list.tolist().index(cf_user)
                user = users.iloc[index]
                delegati = ast.literal_eval(user["Delegati"])
                # print("PROBE:",probe_target,"PAZIENTE:", cf_user, "DELEGATI:",delegati)
                #if the user has delegates
                if len(delegati) > 0 and cf_user != probe_target:
                    if probe_target in delegati:
                        countTG += 1
                    else:
                        countTI += 1
                    max = 0
                    accepted = False
                    for t in delegati:
                        #val = topMatch(probe_template, t, gallery_data,gallery_target, histogram_gallery_data)
                        val = topMatch(t, gallery_target, histogram_gallery_data, hist_probe)
                        # if val > max:
                        #     max = val
                        index_threshold = cf_list.tolist().index(t)
                        if val >= gallery_thresholds[index_threshold]:
                            accepted = True
                            continue
                    # if max >= threshold and probe_target not in delegati:
                    #     fa += 1
                    # elif max < threshold and probe_target in delegati:
                    #     fr += 1
                    if accepted and probe_target not in delegati:
                        fa += 1
                    elif not accepted and probe_target in delegati:
                        fr += 1
                    # print(val, threshold, fr, fa)
        print(probe_target, fr, countTG, fa, countTI)
    return fa, fr, countTG, countTI

if __name__ == '__main__':
    #verificationFRR()
    #verificationFAR()
    verificationROC()
    #evaluationIdentificationAsMultiVer()
    #evaluationIdentification()
