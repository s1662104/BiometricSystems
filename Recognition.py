import cv2
import pandas as pd
import numpy as np
import LBP
import math
import ast
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

#face identification (or 1:N face recognition) consists in finding the identity corresponding to a given face.
def identify(cf, img):

    #upload the various datasets
    gallery_target = np.load("npy_db/gallery_target.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    users = pd.read_csv('dataset_user.csv', index_col=[0])
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    galley_users = list(dict.fromkeys(gallery_target))

    #find the user linked to the cf
    cf_list = users['Codice Fiscale']

    # check if the user is registered
    if cf_list.tolist().__contains__(cf):
        print("UTENTE PRESENTE")
    else:
        print("UTENTE NON PRESENTE! Il codice fiscale inserito è:", cf)
        return None, 0, None

    index = cf_list.tolist().index(cf)
    user = users.iloc[index]

    #find the user's delegates
    delegati = ast.literal_eval(user["Delegati"])

    #if the user doesn't have delegates
    if len(delegati) == 0:
        print("L'utente non ha delegati!")
        return None, 0, None

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
        th_index = galley_users.index(d)
        if val > max  and val >= gallery_thresholds[th_index]:
            max = val
            identity = d    #the identity of the delegate who gets the best value for the moment

    print("L'identità del delegato è:",identity)

    if identity is not None:
        indexd = cf_list.tolist().index(identity)
        recUser = users.iloc[indexd]
        return user, index, recUser
    else:
        return None, 0, None

#face verification (or 1:1 face recognition) consists in checking if a face corresponds to a given identity.
def recognize(cf, img):

    #upload the various datasets
    gallery_target  = np.load("npy_db/gallery_target.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
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
        return user, index, user

    index = cf_list.tolist().index(cf)

    #normalize the input image
    norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    lbp = LBP.Local_Binary_Pattern(1, 8, norm_image)
    hist = lbp.createHistogram(lbp.compute_lbp())

    #calculates the maximum match between the image taken in input and the images in the user's gallery
    val = topMatch(cf, gallery_target, histogram_gallery_data, hist)

    print("val:",val)

    #if the maximum match is greater than or equal to the adaptive threshold, then the identity is verified
    if val >= gallery_thresholds[index]:
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

    #calculate the similarity using the correlation method
    sim = sum1 / math.sqrt(sum_H1*sum_H2)

    return sim

#False Rejection Rate - FRR
#The FRR is defined as the percentage of identification instances in which false rejection occurs.
#This can be expressed as a probability. For example, if the FRR is 0.05 percent, it means that on the average,
#one out of every 2000 authorized persons attempting to access the system will not be recognized by that system.
def verificationFRR():

    pg_target = np.load("npy_db/pg_target.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data =  np.load("npy_db/histogram_pg_data.npy")
    galley_users = list(dict.fromkeys(gallery_target))

    fr = 0

    for i in range(len(pg_target)):

        pg_identity = pg_target[i]
        pg_hist = histogram_pg_data[i]

        #topMatch returns the best match between pg_hist and the templates associated to the claimed identity in the gallery
        gx = topMatch(pg_identity, gallery_target, histogram_gallery_data, pg_hist)
        index = galley_users.index(pg_identity)

        #If the maximum similarity is lower than the threshold, increase the number of False Reject
        if gx < gallery_thresholds[index]:
            fr = fr + 1

    print("TRUE GENUINE:", len(pg_target))
    print("FALSE REJECTION:", fr)
    print("FRR:", fr/len(pg_target))

    return

#False Acceptance Rate - FAR
#The FAR is defined as the percentage of identification instances in which false acceptance occurs.
#This can be expressed as a probability. For example, if the FAR is 0.1 percent, it means that on the average, one out of every 1000
#impostors attempting to breach the system will be successful. Stated another way, it means that the probability of an unauthorized person being identified as
#an authorized person is 0.1 percent.
def verificationFAR():
    pg_target = np.load("npy_db/pg_target.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data = np.load("npy_db/histogram_pg_data.npy")
    histogram_pn_data = np.load("npy_db/histogram_pn_data.npy")
    fa = 0
    ti = 0
    galley_users = list(dict.fromkeys(gallery_target))

    #Scenario in which the impostor doesn't belong to the gallery
    for i in range(len(histogram_pn_data)):
        pn_hist = histogram_pn_data[i]
        index_target = 0
        for t in galley_users:
            #topMatch returns the best match between pg_hist and the templates associated to the claimed identity in the gallery
            val = topMatch(t, gallery_target, histogram_gallery_data, pn_hist)
            if val >= gallery_thresholds[index_target]:
                fa += 1
            ti += 1
            index_target += 1

    #Scenario in which the impostor belongs to the gallery
    for i in range(len(histogram_pg_data)):
        pg_hist = histogram_pg_data[i]
        index_target = 0
        for t in galley_users:
            if t != pg_target[i]:
                # topMatch returns the best match between pg_hist and the templates associated to the claimed identity in the gallery
                val = topMatch(t, gallery_target, histogram_gallery_data, pg_hist)
                if val >= gallery_thresholds[index_target]:
                    fa += 1
                ti += 1
            index_target += 1

    print("TRUE IMPOSTOR:", ti)
    print("FALSE ACCEPTANCE:", fa)
    print("FAR:", fa / ti)

    return

#The Receiver Operating Characteristics (ROC) curve is an evaluation metric for a binary classifier,
#which helps us to visualize the performance of a facial recognition model as its discrimination threshold changes.
#ROC depicts the probability of Genuine Accept (GAR) of the system, expressed as 1-FRR, vs False Accept Rate (FAR) variation.
def verificationROC():
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    pg_target = np.load("npy_db/pg_target.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data = np.load("npy_db/histogram_pg_data.npy")
    histogram_pn_data = np.load("npy_db/histogram_pn_data.npy")
    galley_users = list(dict.fromkeys(gallery_target))

    Y = []
    Y_pred = []

    for i in range(len(histogram_pg_data)):
        Y.append(1)
        index = galley_users.index(pg_target[i])
        val = topMatch(pg_target[i], gallery_target, histogram_gallery_data, histogram_pg_data[i])
        if val >= gallery_thresholds[index]:
            Y_pred.append(1)
        else:
            Y_pred.append(0)

    for j in range(len(histogram_pn_data)):
        index_target = 0
        for t in galley_users:
            Y.append(0)
            val = topMatch(t, gallery_target, histogram_gallery_data, histogram_pn_data[j])
            if val >= gallery_thresholds[index_target]:
                Y_pred.append(1)
            else:
                Y_pred.append(0)
            index_target += 1

    for v in range(len(histogram_pg_data)):
        index_target = 0
        for u in galley_users:
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

def topMatch(identity , gallery_target, histogram_gallery_data, hist):
    max = 0
    index = gallery_target.tolist().index(identity)
    for i in range(5):
        sim = compareHistogram(hist, histogram_gallery_data[index + i])
        if sim >= max:
            max = sim

    return max

def evaluationIdentificationAsMultiVer():
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data = np.load("npy_db/histogram_pg_data.npy")
    histogram_pn_data = np.load("npy_db/histogram_pn_data.npy")
    pg_target = np.load("npy_db/pg_target.npy")
    pn_target = np.load("npy_db/pn_target.npy")
    users = pd.read_csv('dataset_user.csv', index_col=[0])
    cf_list = users['Codice Fiscale']

    results1 = delegatesMatch(histogram_pg_data,pg_target,gallery_target,cf_list,users, gallery_thresholds, histogram_gallery_data)
    results2 = delegatesMatch(histogram_pn_data, pn_target, gallery_target, cf_list, users, gallery_thresholds, histogram_gallery_data)

    print("PG_DATA:",results1)
    print("PN_DATA:", results2)
    print()

    fa = results1[0] + results2[0]
    fr = results1[1] + results2[1]
    countTG = results1[2] + results2[2]
    countTI = results1[3] + results2[3]

    FRR = fr/countTG
    FAR = fa/countTI

    print("FRR:", FRR, countTG)
    print("FAR:", FAR, countTI)

def delegatesMatch(hist_data, target, gallery_target, cf_list, users, gallery_thresholds, histogram_gallery_data):
    countTG = 0
    countTI = 0
    fa = 0
    fr = 0
    galley_users = list(dict.fromkeys(gallery_target))
    print(cf_list[0])
    for i in range(len(hist_data)):
        probe_target = target[i]
        hist_probe = hist_data[i]
        for j in range(len(galley_users)):
                cf_user = galley_users[j]
                index = cf_list.tolist().index(cf_user)
                user = users.iloc[index]
                delegati = ast.literal_eval(user["Delegati"])
                #if the user has delegates
                if len(delegati) > 0 and cf_user != probe_target:
                    if probe_target in delegati:
                        countTG += 1
                    else:
                        countTI += 1
                    accepted = False
                    for t in delegati:
                        val = topMatch(t, gallery_target, histogram_gallery_data, hist_probe)
                        index_threshold = cf_list.tolist().index(t)
                        if val >= gallery_thresholds[index_threshold]:
                            accepted = True
                            continue
                    if accepted and probe_target not in delegati:
                        fa += 1
                    elif not accepted and probe_target in delegati:
                        fr += 1
        print(probe_target, fr, countTG, fa, countTI)
    return fa, fr, countTG, countTI

def evaluationVerification():

    pg_target = np.load("npy_db/pg_target.npy")
    pn_target = np.load("npy_db/pn_target.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data = np.load("npy_db/histogram_pg_data.npy")
    histogram_pn_data = np.load("npy_db/histogram_pn_data.npy")
    galley_identities = list(dict.fromkeys(gallery_target))

    similarity_matrix = np.zeros((len(histogram_pg_data)+len(histogram_pn_data), len(galley_identities)))

    for x in range(len(histogram_pg_data)+len(histogram_pn_data)):
        # Select the histogram probe
        if x < len(histogram_pg_data):
            hist_probe = histogram_pg_data[x]
        else:
            hist_probe = histogram_pn_data[x-len(histogram_pg_data)]

        # Sign for each identity the max similarity against probe
        for y in range(len(galley_identities)):
            claimed_identity = galley_identities[y]
            similarity_matrix[x][y] = topMatch(claimed_identity, gallery_target, histogram_gallery_data, hist_probe)

    np.save("similarity_matrix.npy", similarity_matrix)

    similarity_matrix = np.load("similarity_matrix.npy")

    tg = len(pg_target)
    ti = len(pg_target)*(len(galley_identities)-1) + len(pn_target)*len(galley_identities)
    print("TG:", tg)
    print("TI:", ti)
    print()

    frr = []
    far = []
    gar = []
    grr = []
    thresholds = []

    for threshold in np.arange(0.0, 1.01, 0.01):

        # Fix the threshold
        th = np.round(threshold, 2)

        # Reset all variables
        ga = 0
        fa = 0
        fr = 0
        gr = 0

        # For each probe we take the real identity of the probe and its histogram
        for x in range(similarity_matrix.shape[0]):
            if x < len(histogram_pg_data):
                real_identity = pg_target[x]
            else:
                real_identity = pn_target[x-len(histogram_pg_data)]

            # We claim every identity enrolled in the gallery
            for y in range(similarity_matrix.shape[1]):
                claim_identity = galley_identities[y]
                similarity = similarity_matrix[x][y]

                # If the max similarity is above the threshold:
                if similarity >= th:
                    # If the claimed identity is truw, we increase the GA
                    if real_identity == claim_identity:
                        ga += 1
                    # Otherwise we increase the FA
                    else:
                        fa += 1
                # If the max similarity is below the threshold:
                else:
                    # If the claimed identity is truw, we increase the FR
                    if real_identity == claim_identity:
                        fr += 1
                    # Otherwise we increase the GR
                    else:
                        gr += 1

        gar.append(ga/tg)
        far.append(fa/ti)
        frr.append(fr/tg)
        grr.append(gr/ti)
        thresholds.append(th)

    print("GAR:",gar)
    print("FAR:", far)
    print("FRR:", frr)
    print("GRR:", grr)
    print()

    eer_1 = np.array(far)[np.nanargmin(np.absolute((np.array(frr) - np.array(far))))]
    eer_2 = np.array(frr)[np.nanargmin(np.absolute((np.array(frr) - np.array(far))))]
    eer = (eer_1 + eer_2) / 2

    print("EER:", eer)

    eer_threshold = np.array(thresholds)[np.nanargmin(np.absolute((np.array(frr) - np.array(far))))]

    print("EER Threshold:", eer_threshold)
    print()
    print("Thresholds:", thresholds)

    plt.plot(far, gar)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.ylabel("Genuine Acceptance Rate")
    plt.xlabel("False Acceptance Rate")
    plt.title('Receiver Operating Characteristic')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.show()

    return

def verificationAdaptive():
    pg_target = np.load("npy_db/pg_target.npy")
    pn_target = np.load("npy_db/pn_target.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    histogram_pg_data = np.load("npy_db/histogram_pg_data.npy")
    galley_users = list(dict.fromkeys(gallery_target))
    similarity_matrix = np.load("similarity_matrix.npy")

    tg = len(pg_target)
    ti = len(pg_target) * (len(galley_users) - 1) + len(pn_target) * len(galley_users)

    for decrement in np.arange(0.00, 0.11, 0.01):
        ga = 0
        fa = 0
        fr = 0
        gr = 0
        for x in range(similarity_matrix.shape[0]):
            if x < len(histogram_pg_data):
                real_identity = pg_target[x]
            else:
                real_identity = pn_target[x - len(histogram_pg_data)]
            for y in range(similarity_matrix.shape[1]):
                claim_identity = galley_users[y]
                similarity = similarity_matrix[x][y]
                if similarity >= (gallery_thresholds[y]-decrement):
                    if real_identity == claim_identity:
                        ga += 1
                    else:
                        fa += 1
                else:
                    if real_identity == claim_identity:
                        fr += 1
                    else:
                        gr += 1
        print("Decrementando i threshold adattivi di", decrement,"il FRR è uguale  a:", fr/tg)
        print("Decrementando i threshold adattivi di", decrement,"il FAR è uguale  a:", fa/ti)
        print()

    return

if __name__ == '__main__':
    #verificationFRR()
    #verificationFAR()
    #verificationROC()
    evaluationIdentificationAsMultiVer()
    #verificationAdaptive()
    #evaluationVerification()

    #frr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007142857142857143, 0.007142857142857143, 0.007142857142857143, 0.007142857142857143, 0.007142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02857142857142857, 0.02857142857142857, 0.04285714285714286, 0.05, 0.06428571428571428, 0.07142857142857142, 0.09285714285714286, 0.12857142857142856, 0.15, 0.17142857142857143, 0.17857142857142858, 0.20714285714285716, 0.24285714285714285, 0.2785714285714286, 0.3, 0.34285714285714286, 0.38571428571428573, 0.45, 0.5357142857142857, 0.5785714285714286, 0.6285714285714286, 0.6714285714285714, 0.75, 0.8, 0.8428571428571429, 0.8785714285714286, 0.9214285714285714, 0.9428571428571428, 0.9571428571428572, 0.9857142857142858, 0.9857142857142858, 0.9928571428571429, 0.9928571428571429, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    #far = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9997198879551821, 0.9997198879551821, 0.9991596638655462, 0.9984593837535014, 0.9974789915966387, 0.9957983193277311, 0.9928571428571429, 0.9882352941176471, 0.9831932773109243, 0.9746498599439776, 0.9644257703081233, 0.9511204481792717, 0.9329131652661065, 0.9119047619047619, 0.8813725490196078, 0.846078431372549, 0.8081232492997199, 0.7675070028011205, 0.7177871148459384, 0.6668067226890756, 0.6095238095238096, 0.5497198879551821, 0.49089635854341734, 0.426750700280112, 0.3680672268907563, 0.315266106442577, 0.26554621848739496, 0.22198879551820727, 0.18207282913165265, 0.14523809523809525, 0.11708683473389356, 0.09243697478991597, 0.07226890756302522, 0.05742296918767507, 0.044677871148459385, 0.033473389355742296, 0.025070028011204483, 0.018067226890756304, 0.012324929971988795, 0.008963585434173669, 0.0056022408963585435, 0.0036414565826330533, 0.0021008403361344537, 0.0016806722689075631, 0.0014005602240896359, 0.0004201680672268908, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #thresholds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]

    # index = thresholds.index(0.62)
    # print("Valori ottenuti utilizzando il eer_threshold = 0.62")
    # print("FRR:", frr[index])
    # print("FAR:", far[index])
    # index = far.index(0.0036414565826330533)
    # print("Valori ottenuti utilizzando un threshold =", thresholds[index], "che ci dia un far simile a quello ottenuto con i threshold adattivi")
    # print("FRR:", frr[index])
    # print("FAR:", far[index])

