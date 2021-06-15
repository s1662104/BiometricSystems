import cv2
import pandas as pd
import numpy as np
import LBP
import math
import ast
import os.path
from matplotlib import pyplot as plt

# ritorna i dati del paziente cf e del delgato
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

# restituisce il numero di False Accept, False Reject, True Genuine e True Impostor
def delegatesMatch(hist_data, target, gallery_target, cf_list, users, gallery_thresholds, histogram_gallery_data):

    # inizializzazione delle variabili
    countTG = 0
    countTI = 0
    fa = 0
    fr = 0
    galley_users = list(dict.fromkeys(gallery_target))

    # per ogni istogramma
    for i in range(len(hist_data)):
        probe_target = target[i]        #identità a cui appartiene l'istogramma
        hist_probe = hist_data[i]       #istogramma
        # per ogni utente della gallery
        for j in range(len(galley_users)):
                cf_user = galley_users[j]       #codice fiscale dell'utente
                index = cf_list.tolist().index(cf_user)
                user = users.iloc[index]        #informazioni dell'utente
                delegati = ast.literal_eval(user["Delegati"])       #lista dei delegati dell'utente
                # se l'utente ha dei delegati e se l'identità dell'utente non corrisponde a quella del probe
                if len(delegati) > 0 and cf_user != probe_target:
                    # se il probe è un delegato, allora è un True Genuine
                    if probe_target in delegati:
                        countTG += 1
                    # altrimenti è un True Impostor
                    else:
                        countTI += 1
                    accepted = False
                    # per ogni delegato
                    for t in delegati:
                        # restituisce la migliore corrispondenza tra hist_probe e gli istogrammi del delegato nella galleria
                        val = topMatch(t, gallery_target, histogram_gallery_data, hist_probe)
                        index_threshold = cf_list.tolist().index(t)
                        # se la similarità supera il threshold adattivo, allora viene riconosciuto come uno dei delegati
                        if val >= gallery_thresholds[index_threshold]:
                            accepted = True
                            continue
                    # se il probe viene riconosciuto ma non è un delegato, incremente False Accept
                    if accepted and probe_target not in delegati:
                        fa += 1
                    # se il probe non viene riconosciuto nonostante sia un delegato, incremente False Reject
                    elif not accepted and probe_target in delegati:
                        fr += 1
        print(probe_target, fr, countTG, fa, countTI)
    return fa, fr, countTG, countTI

# mostra i False Rejection Rate, i False Acceptance Rate, l'Equal Error Rate e la ROC curve nel caso in cui si adottano i threshold fissi
def evaluationVerification():

    # caricamento dei vari dataset
    pg_target = np.load("npy_db/pg_target.npy")
    pn_target = np.load("npy_db/pn_target.npy")
    gallery_target = np.load("npy_db/gallery_target.npy")
    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
    histogram_pg_data = np.load("npy_db/histogram_pg_data.npy")
    histogram_pn_data = np.load("npy_db/histogram_pn_data.npy")
    galley_identities = list(dict.fromkeys(gallery_target))

    # se non esiste un file contenente la similarity matrix, allora viene creato
    if not os.path.exists("similarity_matrix.npy"):
        # creazione della similarity matrix
        similarity_matrix = np.zeros((len(histogram_pg_data)+len(histogram_pn_data), len(galley_identities)))
        # per ogni istogramma ottenuto dai probe
        for x in range(len(histogram_pg_data)+len(histogram_pn_data)):
            # seleziona l'istogramma del probe gallery
            if x < len(histogram_pg_data):
                hist_probe = histogram_pg_data[x]
            # seleziona l'istogramma del probe non appartenete alla gallery
            else:
                hist_probe = histogram_pn_data[x-len(histogram_pg_data)]
            # segna per ogni identità la massima somiglianza con il probe
            for y in range(len(galley_identities)):
                claimed_identity = galley_identities[y]
                similarity_matrix[x][y] = topMatch(claimed_identity, gallery_target, histogram_gallery_data, hist_probe)
        # salva la similarity matrix
        np.save("similarity_matrix.npy", similarity_matrix)
    # se il file esiste, la matrice è stata già creata e quindi viene caricata
    else:
        similarity_matrix = np.load("similarity_matrix.npy")
    # calcolo dei True Genuine e dei True Impostors
    tg = len(pg_target)
    ti = len(pg_target)*(len(galley_identities)-1) + len(pn_target)*len(galley_identities)
    print("TG:", tg)
    print("TI:", ti)
    print()

    # inizializzazione delle variabili che conterranno i vari rate e thresholds
    frr = []
    far = []
    gar = []
    grr = []
    thresholds = []

    # per ogni threshold
    for threshold in np.arange(0.0, 1.01, 0.01):

        # si fissa la soglia
        th = np.round(threshold, 2)

        # si resettano le variabili
        ga = 0
        fa = 0
        fr = 0
        gr = 0

        # per ogni probe prendiamo l'identità reale e il suo istogramma (ogni riga corrisponde ad un probe)
        for x in range(similarity_matrix.shape[0]):
            if x < len(histogram_pg_data):
                real_identity = pg_target[x]
            else:
                real_identity = pn_target[x-len(histogram_pg_data)]

            # per ogni identità iscritta alla gallery (ogni colonna corrisponde ad un utente)
            for y in range(similarity_matrix.shape[1]):
                claim_identity = galley_identities[y]
                similarity = similarity_matrix[x][y]

                # se la similairty massima è al di sopra della soglia
                if similarity >= th:
                    # e se l'identità dichiarata è vera, aumenta Genuine Accept
                    if real_identity == claim_identity:
                        ga += 1
                    # altrimenti incrementa  False Accept
                    else:
                        fa += 1
                # se la similairty massima è inferiore alla soglia
                else:
                    # e se l'identità dichiarata è vera, aumenta False Reject
                    if real_identity == claim_identity:
                        fr += 1
                    # altrimenti incrementa Genuine Reject
                    else:
                        gr += 1
        # aggiungiamo i rate relativi al threshold th alle varie liste
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

    # cacolo dell'Equal Error Rate
    eer_1 = np.array(far)[np.nanargmin(np.absolute((np.array(frr) - np.array(far))))]
    eer_2 = np.array(frr)[np.nanargmin(np.absolute((np.array(frr) - np.array(far))))]
    eer = (eer_1 + eer_2) / 2
    print("EER:", eer)
    # cacolo del threshold dell'Equal Error Rate
    eer_threshold = np.array(thresholds)[np.nanargmin(np.absolute((np.array(frr) - np.array(far))))]
    print("EER Threshold:", eer_threshold)
    print()

    print("Thresholds:", thresholds)

    # traccia la ROC curve
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
    evaluationIdentificationAsMultiVer()
