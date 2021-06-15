import cv2
import pandas as pd
import numpy as np
import LBP
import math
import ast
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# restituisce i dati del delegato e del cliente cf utilizzando img per il confronto
def identify(cf, img):

    # caricamento dei vari dataset
    gallery_target = np.load("npy_db/gallery_target.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    users = pd.read_csv('dataset_user.csv', index_col=[0])
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    galley_users = list(dict.fromkeys(gallery_target))
    cf_list = users['Codice Fiscale']

    # informazioni del paziente utilizzando cf
    index = cf_list.tolist().index(cf)
    user = users.iloc[index]

    # lista dei delegati del paziente
    delegati = ast.literal_eval(user["Delegati"])

    # se il paziente non ha nessun delegato, allora termina l'operazione
    if len(delegati) == 0:
        print("L'utente non ha delegati!")
        return None, 0, None

    # si inizializzano le variabili per la similarity e l'identità del delegato
    max = 0
    identity = None

    # l'immagine in input viene normalizzata per poter utilizzare LBP e ottenerne l'istogramma relativo
    norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    lbp = LBP.Local_Binary_Pattern(1, 8, norm_image)
    hist = lbp.createHistogram(lbp.compute_lbp())

    # per ogni delegato
    for d in delegati:
        # ottengo il miglior valore ottenuto confrontando l'istograma con quelli del delegato nella galleria
        val = topMatch(d, gallery_target, histogram_gallery_data, hist)
        th_index = galley_users.index(d)
        # confrontiamo la similarity con quella massimo ottenuta finora e con il threshold del delegato
        # se la similarity supera quella massima e il threshold, allora aggiorniamo le variabili
        if val > max  and val >= gallery_thresholds[th_index]:
            max = val       # il più alto valore di similarity attuale
            identity = d    # identità del delegato che ha ottenuto il miglior valore per il momento

    # se c'è stato un riconoscimento tra i delegati
    if identity is not None:
        indexd = cf_list.tolist().index(identity)
        recUser = users.iloc[indexd]
        # ritorna i dati del paziente, l'indice in cui si trova in dataset_user e le informazioni del delegato
        return user, index, recUser
    else:
        # altrimenti ritorna None come paziente, 0 come indice e None come delegato
        return None, 0, None

# restituisce l'identità del paziente cf se img dà un riscontro positivo
def recognize(cf, img):

    # caricamento dei vari dataset
    gallery_target  = np.load("npy_db/gallery_target.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    users = pd.read_csv('dataset_user.csv', index_col=[0])
    cf_list = users['Codice Fiscale']

    # inizializzazione delle variabili che conterranno i dati del paziente e l'indice in cui si trova in dataset_user
    user = None
    index = cf_list.tolist().index(cf)

    # l'immagine in input viene normalizzata per poter utilizzare LBP e ottenerne l'istogramma relativo
    norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    lbp = LBP.Local_Binary_Pattern(1, 8, norm_image)
    hist = lbp.createHistogram(lbp.compute_lbp())

    # calcola la corrispondenza massima tra l'istogramma dell'immagine in input e quelli delle immagini nella galleria dell'utente
    val = topMatch(cf, gallery_target, histogram_gallery_data, hist)

    # se la similairtà massima è maggiore o uguale alla soglia adattativa del paziente, l'identità viene verificata
    if val >= gallery_thresholds[index]:
        user = users.iloc[index]

    # ritorna le informazioni del paziente e l'indice in cui si trova in dataset_user
    return user, index, user

# ritorna il valore di similarity tra i due istogrammi presi in input
def compareHistogram(H1, H2):

    # per fare la differenza tra i due istogrammi, dobbiamo essere sicuri che la loro lunghezza sia la stessa
    if len(H1) != len(H2):
        print("LA LUNGHEZZA DEI DUE ISTOGRAMMI NON E' LA STESSA")
        return

    # calcolare la media dei valori nei due istogrammi
    avg_H1 = sum(H1) / len(H1)
    avg_H2 = sum(H2) / len(H2)

    # inizializzazione delle variabili
    sum1 = 0
    sum_H1 = 0
    sum_H2 = 0

    for i in range(0, len(H1)):
        sum1 = sum1 + ((H1[i]-avg_H1)*(H2[i]-avg_H2))
        sum_H1 = sum_H1 + pow(H1[i]-avg_H1, 2)
        sum_H2 = sum_H2 + pow(H2[i]-avg_H2, 2)

    # calcolare la somiglianza utilizzando il metodo di correlazione
    sim = sum1 / math.sqrt(sum_H1*sum_H2)

    return sim

# calcola il False Rejection Rate nel caso in cui si utilizzano i threshold adattivi
def verificationFRR():

    # caricamento dei vari dataset
    pg_target = np.load("npy_db/pg_target.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data =  np.load("npy_db/histogram_pg_data.npy")
    galley_users = list(dict.fromkeys(gallery_target))

    # inizializzazione della variabile dei False Rejections
    fr = 0


    for i in range(len(pg_target)):
        pg_identity = pg_target[i]      # identità dell'utente a cui appartiene l'istogramma da controntare
        pg_hist = histogram_pg_data[i]      # istogramma dell'utente

        # restituisce la migliore corrispondenza tra pg_hist e gli istogrammi dell'utente nella galleria
        gx = topMatch(pg_identity, gallery_target, histogram_gallery_data, pg_hist)
        index = galley_users.index(pg_identity)

        # se la somiglianza massima è inferiore alla soglia dell'utente, aumenta il numero di False Reject
        if gx < gallery_thresholds[index]:
            fr = fr + 1

    print("TRUE GENUINE:", len(pg_target))
    print("FALSE REJECTION:", fr)
    print("FRR:", fr/len(pg_target))

    return

# calcola il False Acceptance Rate nel caso in cui si utilizzano i threshold adattivi
def verificationFAR():

    # caricamento dei vari dataset
    pg_target = np.load("npy_db/pg_target.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data = np.load("npy_db/histogram_pg_data.npy")
    histogram_pn_data = np.load("npy_db/histogram_pn_data.npy")
    galley_users = list(dict.fromkeys(gallery_target))

    # inizializzazione della variabile dei False Accept e True Impostor
    fa = 0
    ti = 0

    # scenario in cui l'impostore non appartiene alla galleria
    for i in range(len(histogram_pn_data)):
        pn_hist = histogram_pn_data[i]      # istogramma appartenente ad un impostore
        index_target = 0
        # per ogni utente nella gallery
        for t in galley_users:
            # restituisce la migliore corrispondenza tra pn_hist e gli istogrammi dell'utente nella galleria
            val = topMatch(t, gallery_target, histogram_gallery_data, pn_hist)

            # se il valore è mmaggiore della soglia adattive di un utente, allora si incrementa il False Accept
            if val >= gallery_thresholds[index_target]:
                fa += 1
            ti += 1
            index_target += 1

    # scenario in cui l'impostore appartiene alla galleria
    for i in range(len(histogram_pg_data)):
        pg_hist = histogram_pg_data[i]      # istogramma appartenente ad un utente che dichiara un'altra identità
        index_target = 0
        # per ogni utente nella gallery
        for t in galley_users:
            if t != pg_target[i]:
                # restituisce la migliore corrispondenza tra pg_hist e gli istogrammi dell'utente nella galleria
                val = topMatch(t, gallery_target, histogram_gallery_data, pg_hist)

                # se il valore è mmaggiore della soglia adattive di un utente, allora si incrementa il False Accept
                if val >= gallery_thresholds[index_target]:
                    fa += 1
                ti += 1
            index_target += 1

    print("TRUE IMPOSTOR:", ti)
    print("FALSE ACCEPTANCE:", fa)
    print("FAR:", fa / ti)

    return

# ritorna il massimo valore di similarity tra hist con gli istogrammi appartenenti a identity
def topMatch(identity , gallery_target, histogram_gallery_data, hist):

    # inizializzazione delle variabili che contengono il valore più alto di similarity e l'indice in cui si trova identity in gallery_target
    max = 0
    index = gallery_target.tolist().index(identity)

    for i in range(5):
        sim = compareHistogram(hist, histogram_gallery_data[index + i])
        # se la similairy ottenuta è mmaggiore rispetto al valore più alto calcolato finora, allora si sovrascrive
        if sim >= max:
            max = sim

    return max

# mostra il False Rejection Rate e il False Acceptance Rate nell'operazione di riconoscimento del delegato
def evaluationIdentificationAsMultiVer():

    # caricamento dei vari dataset
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_thresholds = np.load("npy_db/gallery_thresholds.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data = np.load("npy_db/histogram_pg_data.npy")
    histogram_pn_data = np.load("npy_db/histogram_pn_data.npy")
    pg_target = np.load("npy_db/pg_target.npy")
    pn_target = np.load("npy_db/pn_target.npy")
    users = pd.read_csv('dataset_user.csv', index_col=[0])
    cf_list = users['Codice Fiscale']

    # ottingo i risultati dei vari fa, fr, tg, ti con gli istogrammi degli utenti e degli impostori
    results1 = delegatesMatch(histogram_pg_data, pg_target, gallery_target, cf_list, users, gallery_thresholds, histogram_gallery_data)
    results2 = delegatesMatch(histogram_pn_data, pn_target, gallery_target, cf_list, users, gallery_thresholds, histogram_gallery_data)

    print("PG_DATA:",results1)
    print("PN_DATA:", results2)
    print()

    fa = results1[0] + results2[0]          # False Accept totali
    fr = results1[1] + results2[1]          # False Rejection totali
    countTG = results1[2] + results2[2]     # True Genuine totali
    countTI = results1[3] + results2[3]     # True Impostor totali

    FRR = fr/countTG    # False Rejection Rate
    FAR = fa/countTI    # False Acceptance Rate

    print("FRR:", FRR, countTG)
    print("FAR:", FAR, countTI)

def delegatesMatch(hist_data, target, gallery_target, cf_list, users, gallery_thresholds, histogram_gallery_data):

    # inizializzazione delle variabili
    countTG = 0
    countTI = 0
    fa = 0
    fr = 0
    galley_users = list(dict.fromkeys(gallery_target))

    # per ogni istogramma
    for i in range(len(hist_data)):
        probe_target = target[i]
        hist_probe = hist_data[i]
        for j in range(len(galley_users)):
                cf_user = galley_users[j]
                index = cf_list.tolist().index(cf_user)
                user = users.iloc[index]
                delegati = ast.literal_eval(user["Delegati"])

                # se l'utente ha dei delegati
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
        # seleziona l'istogramma del probe
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

    # per ogni threshold
    for threshold in np.arange(0.0, 1.01, 0.01):

        # si fissa il threshold
        th = np.round(threshold, 2)

        # si resettano le variabili
        ga = 0
        fa = 0
        fr = 0
        gr = 0

        # per ogni probe prendiamo l'identità reale e il suo istogramma
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

