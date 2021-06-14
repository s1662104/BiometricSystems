import dlib
from enum import Enum
import cv2
import Recognition
import tkinter as tk
import tkinter.messagebox as messagebox
from PIL import ImageTk, Image
import numpy as np
import pandas as pd
import ast
from datetime import date
from LBP import Local_Binary_Pattern

from EyeBlink import EyeBlink
from MicroTexture import MicroTexture

messageCF = "Inserire codice fiscale: "
messageError = "Input non valido"
messageN = "Inserire nome: "
choice1 = "Registrazione"
choice2 = "Prelievo Farmaci"
messageWelcome = "Benvenuto\n Che operazione desideri svolgere?"
numberMedicines = "Quanti farmaci assumi?"
numberDelegate = "A chi vuoi delegare il prelievo di farmaci?"
enrollmentCompleted = "REGISTRAZIONE COMPLETATA!"
recognitionRejected = "UTENTE NON RICONOSCIUTO"
messageRecognition = "Chi sei?"
recognitionChoice1 = "Paziente"
recognitionChoice2 = "Delegato"
spoofingMessage = "ANTISPOOFING ERROR!\n L'UTENTE NON SEMBRA ESSERE REALE!"
messageMedicineError = "Inserisci i farmaci"
messageDelegateError = "Inserisci i codici fiscali corretti dei tuoi delegati"

dim_image = 64
number_maximum_delegate = 3
n_photo_x_user = 5
pages = Enum("pages", [])
radius = 1
neighborhood = 8


class Page(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        global pages
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        self.frames = {}
        pageNames = []

        for F in (StartPage, EnrollmentPage, RecognitionPage, DataEnrollmentPage, DataRecognitionPage,
                  InformationPage, UserPage, RecognitionChoicePage):
            frame = F(container, self)

            self.frames[F] = frame
            pageNames.append(F.__name__)

            frame.grid(row=0, column=0, sticky="nsew")

        # utilizzo gli enum per indicizzare in modo piu' leggibile alle pagine. Quindi piuttosto che puntare alla
        # pagina n, si indicizza a pages.NOME_PAGINA.value - 1 (-1 perche' gli indici partono da 1)
        pages = Enum("pages", pageNames)

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


# Pagina iniziale dove si presentano le due scelte: registrazione oppure prelievo farmaci
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=messageWelcome)
        label.pack(pady=10, padx=50)
        button1 = tk.Button(self, text=choice1, width=15, height=2, bg='#1E79FA',
                            command=lambda: goToEnroll())
        button2 = tk.Button(self, text=choice2, width=15, height=2, bg='#1E79FA',
                            command=lambda: goToRecognize())
        button1.pack()
        button2.pack(pady=1)

        def goToEnroll():
            # resetto la pagina da eventuali dati inseriti da un altro utente. Non lo faccio dopo essere passati
            # da questa alla pagina successiva perche' l'utente potrebbe sempre tornare indietro a correggere i dati
            list(controller.frames.values())[pages.EnrollmentPage.value - 1].reset()
            controller.show_frame(EnrollmentPage)

        def goToRecognize():
            controller.show_frame(RecognitionChoicePage)


# pagina di registrazione
class EnrollmentPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=choice1)
        label.pack(pady=10, padx=10)

        self.entryCF = tk.Entry(self)
        self.entryCF.insert(1, messageCF)
        self.entryCF.pack(pady=2)
        self.entryName = tk.Entry(self)
        self.entryName.insert(1, messageN)
        self.entryName.pack(pady=2)
        button2 = tk.Button(self, text="Invia", width=10, height=1, bg='#1E79FA',
                            command=lambda: check_input(controller, self.entryCF.get(), labelError, 0, None,
                                                        self.entryName.get()))
        button2.pack()

        labelError = tk.Label(self, text=messageError, fg="#f0f0f0")
        labelError.pack(pady=10, padx=10)

        tk.Button(self, text="Indietro", width=8, height=1, bg='#1E79FA',
                  command=lambda: back(controller, self.entryCF, labelError, self.entryName)).pack(side="left",
                                                                                                   pady=300)

    # reset dei campi della pagina
    def reset(self):
        self.entryCF.delete(0, tk.END)
        self.entryCF.insert(0, messageCF)
        self.entryName.delete(0, tk.END)
        self.entryName.insert(0, messageN)


# Qui l'utente deve scegliere se essere riconosciuto paziente o come delegato
class RecognitionChoicePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=messageRecognition)
        label.pack(pady=10, padx=50)
        button1 = tk.Button(self, text=recognitionChoice1, width=15, height=2, bg='#1E79FA',
                            command=lambda: confirm(0))
        button2 = tk.Button(self, text=recognitionChoice2, width=15, height=2, bg='#1E79FA',
                            command=lambda: confirm(1))
        button1.pack()
        button2.pack(pady=1)

        tk.Button(self, text="Indietro", width=8, height=1, bg='#1E79FA',
                  command=lambda: controller.show_frame(StartPage)).place(y=520, x=2)

        # resetto la pagina e aggiorno il ruolo dell'utente (paziente o delegato)
        def confirm(role):
            list(controller.frames.values())[pages.RecognitionPage.value - 1].reset()
            list(controller.frames.values())[pages.RecognitionPage.value - 1].update_data(role)
            controller.show_frame(RecognitionPage)


# pagina per il riconoscimento dell'utente
class RecognitionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=choice2)
        label.pack(pady=10, padx=10)

        self.role = 0

        self.entryCF = tk.Entry(self)
        self.entryCF.insert(1, messageCF)
        self.entryCF.pack(padx=0, pady=0)
        button2 = tk.Button(self, text="Invia", width=10, height=1, bg='#1E79FA',
                            command=lambda: check_input(controller, self.entryCF.get(), labelError, 1, self.role))
        button2.pack()

        labelError = tk.Label(self, text=messageError, fg="#f0f0f0")
        labelError.pack(pady=10, padx=10)

        tk.Button(self, text="Indietro", width=8, height=1, bg='#1E79FA',
                  command=lambda: back(controller, self.entryCF, labelError)).pack(side="left", pady=385)

    # reset della pagina
    def reset(self):
        self.entryCF.delete(0, tk.END)
        self.entryCF.insert(0, messageCF)

    # aggiornamento della pagina
    def update_data(self, role):
        self.role = role


# La pagina serve per mostrare i dati dell'utente in caso di registrazione o di riconoscimento.
class DataPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.photo = np.ones((64, 64)) * 150
        img = ImageTk.PhotoImage(image=Image.fromarray(self.photo))
        self.panel = tk.Label(self, image=img)
        self.panel.image = img
        self.panel.pack(pady=10, padx=10)

        self.name = tk.Label(self, text="UTENTE RICONOSCIUTO")
        self.name.pack()

        self.patient = tk.Label(self, text="PAZIENTE")
        self.patient.pack()

        self.cf = tk.Label(self, text="CF")
        self.cf.pack()


# Dati dell'utente nella fase di registrazione
class DataEnrollmentPage(DataPage):
    def __init__(self, parent, controller):
        DataPage.__init__(self, parent, controller)

        self.patient.destroy()

        labelDelegate = tk.Label(self, text=numberDelegate)
        labelDelegate.pack()
        self.delegateEntry = []
        for n in range(number_maximum_delegate):
            entryDelegate = tk.Entry(self)
            entryDelegate.insert(1, "")
            entryDelegate.pack(padx=0, pady=0)
            self.delegateEntry.append(entryDelegate)

        self.medicineEntry = []
        labelNMedicine = tk.Label(self, text=numberMedicines)
        labelNMedicine.pack()
        self.entryNMedicine = tk.Entry(self)
        self.entryNMedicine.insert(1, "")
        self.entryNMedicine.pack(padx=0, pady=0)
        button = tk.Button(self, text="Invia", width=10, height=1, bg='#1E79FA',
                           command=lambda: self.addMedicines(self.entryNMedicine.get()))
        button.pack()

        tk.Button(self, text="Conferma", width=8, height=1, bg='#1E79FA',
                  command=lambda: self.confirm(controller)).place(y=520, x=220)

        tk.Button(self, text="Indietro", width=8, height=1, bg='#1E79FA',
                  command=lambda: self.back(controller)).place(y=520, x=2)

    # resetta la pagina
    def reset(self):
        self.entryNMedicine.delete(0, tk.END)
        self.entryNMedicine.insert(0, "")
        self.addMedicines("0")
        for i in range(number_maximum_delegate):
            self.delegateEntry[i].delete(0, tk.END)
            self.delegateEntry[i].insert(0, "")

    # aggiorna i dati della pagina
    def update_data(self, cf, img, photo, name):
        self.name.config(text="NOME: " + name)
        self.cf.config(text="CODICE FISCALE: " + cf)
        self.panel.config(image=img)
        self.panel.image = img
        self.photo = photo

    # aggiunge i campi compilabili in base al numero di farmaci indicato
    def addMedicines(self, n_medicine):
        if not n_medicine.isdigit():
            self.bell()
        else:
            n = int(n_medicine)
            actualN = len(self.medicineEntry)
            if n > actualN:
                n = n - actualN
                for m in range(n):
                    entryMedicine = tk.Entry(self)
                    entryMedicine.insert(1, "")
                    entryMedicine.pack(pady=1, padx=30)
                    self.medicineEntry.append(entryMedicine)
            elif n < actualN:
                n = actualN - n
                for m in range(n):
                    entryMedicine = self.medicineEntry[len(self.medicineEntry) - 1]
                    entryMedicine.destroy()
                    self.medicineEntry.pop()

    # ritorna alla pagina precedente, ma non resetta perche' potrebbe essere che l'utente abbia sbagliato i dati
    # compilati
    def back(self, controller):
        controller.show_frame(EnrollmentPage)

    # conferma le scelte dell'utente
    def confirm(self, controller):
        medicineError = False
        # se non sono stati inseriti farmaci
        if len(self.medicineEntry) == 0:
            medicineError = True
        else:
            # se un campo e' vuoto
            for medicine in self.medicineEntry:
                if medicine.get() == "":
                    medicineError = True
                    break
        # allora mostra un messaggio di errore
        if medicineError:
            messagebox.showerror(title="Errore", message=messageMedicineError)
        else:
            delegatesError = False
            delegates = []
            # controlla i delegati
            for delegate in self.delegateEntry:
                if delegate.get() != "":
                    # controlla se il codice fiscale e' valido
                    if not isCF(delegate.get()):
                        delegatesError = True
                        break
                    else:
                        delegates.append(delegate.get())
            # altrimenti mostra un messaggio d'errore
            if delegatesError:
                messagebox.showerror(title="Errore", message=messageDelegateError)
            else:
                medicines = []
                # ottiene la lista delle medicine
                for medicine in self.medicineEntry:
                    medicines.append(medicine.get())
                # aggiunge l'utente al sistema
                addUser(self.photo, self.cf.cget("text")[16:], self.name.cget("text")[6:], medicines, delegates)
                list(controller.frames.values())[pages.InformationPage.value - 1].update_data(enrollmentCompleted)
                controller.show_frame(InformationPage)


# Mostra i dati dell'utente prossimo al riconoscimento
class DataRecognitionPage(DataPage):
    def __init__(self, parent, controller):
        DataPage.__init__(self, parent, controller)

        self.name.destroy()
        self.patient.destroy()
        self.role = 0

        tk.Button(self, text="Conferma", width=8, height=1, bg='#1E79FA',
                  command=lambda: self.confirm(controller)).place(y=520, x=220)

        tk.Button(self, text="Indietro", width=8, height=1, bg='#1E79FA',
                  command=lambda: self.back(controller)).place(y=520, x=2)

    # reset della pagina
    def reset(self):
        pass

    # aggiornamento della pagina
    def update_data(self, cf, img, photo, role, name=None):
        self.role = role
        self.cf.config(text="CODICE FISCALE: " + cf)
        self.panel.config(image=img)
        self.panel.image = img
        self.photo = photo

    # si passa alla pagina successiva
    def confirm(self, controller):
        # se si tratta di Verifica del paziente
        if self.role == 0:
            print("Verifica del paziente")
            patient, index, user = Recognition.recognize(self.cf.cget("text")[16:], self.photo)
        # se si tratta di Verifica del delegato
        else:
            print("Verifica del delegato")
            patient, index, user = Recognition.identify(self.cf.cget("text")[16:], self.photo)
        print("Il paziente e':", patient, "L'utente riconosciuto e':", user)
        if user is not None:
            list(controller.frames.values())[pages.UserPage.value - 1].reset()
            list(controller.frames.values())[pages.UserPage.value - 1].update_data(index, user["User"], patient["User"],
                                                                                   patient["Codice Fiscale"],
                                                                                   patient["Delegati"], patient["Farmaci"],
                                                                                   patient["Data"],
                                                                                   self.panel.image)
            controller.show_frame(UserPage)
        else:
            list(controller.frames.values())[pages.InformationPage.value - 1].update_data(recognitionRejected)
            controller.show_frame(InformationPage)

    # ritorna alla pagina precedente
    def back(self, controller):
        controller.show_frame(RecognitionPage)


# pagina che mostra delle informazioni,ad esempio se il riconoscimento o il test anti-spoofing non e' andato a buno fine
class InformationPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.label = tk.Label(self, text="")
        self.label.pack(pady=200)

        tk.Button(self, text="Home", width=8, height=1, bg='#1E79FA',
                  command=lambda: controller.show_frame(StartPage)).place(y=520, x=110)

    def update_data(self, info):
        self.label.config(text=info)


# pagina in cui vengono mostrati i dati del paziente
class UserPage(DataPage):
    def __init__(self, parent, controller):
        DataPage.__init__(self, parent, controller)

        delegates = tk.Label(self, text="DELEGATI:")
        delegates.pack()
        self.delegatesLabels = []
        for i in range(3):
            delegate = tk.Label(self, text="Delegato")
            delegate.pack()
            self.delegatesLabels.append(delegate)
        medicineLabel = tk.Label(self, text="FARMACI:")
        medicineLabel.pack()

        self.entries = []

        tk.Button(self, text="Home", width=8, height=1, bg='#1E79FA',
                  command=lambda: controller.show_frame(StartPage)).place(y=520, x=110)

    # resetta la pagina
    def reset(self):
        for i, label in enumerate(self.delegatesLabels):
            self.delegatesLabels[i].config(text="-")
        for entry in self.entries:
            entry.destroy()

    # aggiorna i dati
    def update_data(self, index, rec_user, patient, cf, delegates, medicines, last_date, photo):
        self.cf.config(text="CODICE FISCALE: " + cf)
        self.name.config(text="UTENTE RICONOSCIUTO: " + rec_user)
        self.patient.config(text="PAZIENTE: " + patient)
        self.panel.config(image=photo)
        self.panel.image = photo
        # aggiorna la lista dei delegati
        delegates = ast.literal_eval(delegates)
        for i, label in enumerate(self.delegatesLabels):
            if i < len(delegates):
                self.delegatesLabels[i].config(text=delegates[i])
            else:
                self.delegatesLabels[i].config(text="-")
        # aggiorna la lista dei farmaci
        medicines = ast.literal_eval(medicines)
        for medicine in medicines:
            label = tk.Label(self, text=medicine)
            label.pack()
            self.entries.append(label)
        # indica la lista dei farmaci prelevati
        label = tk.Label(self, text="FARMACI PRELEVATI:")
        label.pack()
        self.entries.append(label)
        self.obtainable_medicines(last_date, medicines)
        # aggiorna la data dell'ultimo prelievo
        csv = pd.read_csv("dataset_user.csv", index_col=[0])
        csv.iloc[index]["Data"] = date.today().strftime("%d/%m/%Y")
        csv.to_csv('dataset_user.csv')

    # calcola la lista dei farmaci prelevabili
    def obtainable_medicines(self, d: str, medicines):
        dmy = d.split("/")
        # ottengo la data
        last_date = date(int(dmy[2]), int(dmy[1]), int(dmy[0]))
        # prendo la data di oggi
        today = date.today()
        # calcolo il numero di giorni che sono passati
        days = (today - last_date).days
        medicine_dataset = pd.read_csv("dataset_medicine.csv")
        for row in medicine_dataset.iterrows():
            for medicine in medicines:
                # poiche' le medicine sono nel formato "Prefolic 15 mg", si divide il nome dal dosaggio
                s = medicine.split(" ")
                # gli ultimi due elementi sono il dosaggio ("15 mg")
                indices = len(s) - 2
                # si fa il join di tutti gli elementi dell'array, eccetto gli ultimi due
                name = " ".join([s[i] for i in range(indices)])
                # si prendono gli ultimi due elementi
                dose = s[len(s) - 2] + " " + s[len(s) - 1]
                # se il nome e il dosaggio combaciano
                if row[1]["Nome"] == name and row[1]["Dosaggio"] == dose:
                    # estraggo i dati
                    n = row[1]["Numero Pasticche"]
                    dose_x_day = row[1]["Dose x giorno"]
                    # calcolo il numero di scatole in base al pasticche in una confezione e alle dosi assunte in un
                    # giorno
                    box = int(days / n) * dose_x_day
                    label = tk.Label(self, text=medicine + " x " + str(box))
                    label.pack()
                    self.entries.append(label)


# funziona chiamata da EnrollmentPage e RecognitionPage per verificare se l'input fornito e' corretto
def check_input(controller, cf, label_error, op, role=None, name=None):
    if not isCF(cf) or cf == messageCF or (name is not None and name == messageN):
        print("? Errore")
        label_error.configure(fg="red")
        return
    else:
        label_error.configure(fg="#f0f0f0")
    if op == 0:
        n = pages.DataEnrollmentPage.value - 1
    else:
        n = pages.DataRecognitionPage.value - 1
    list(controller.frames.values())[n].reset()
    # Inizio parte antispoofing in fase di matching
    nameFileCsv = 'histogram.csv'
    # l'utente deve passare entrambi i test di anti-spoofing
    if not EyeBlink(None).eyeBlinkStart():
        user = False
    elif not MicroTexture(nameFileCsv).microTextureCam():
        user = False
    else:
        user = True
    # Fine parte antispoofing in fase di matching
    # registrazione
    if user:
        if op == 0:
            cropList = multipleCapture()
            # list(controller.frames.values())[n].update_data(cf, ImageTk.PhotoImage(image=Image.fromarray(crop)), crop,
            #                                                name)
            # si passa una solo immagine, come immagine rappresentativa dell'utente
            list(controller.frames.values())[n].update_data(cf, ImageTk.PhotoImage(image=Image.fromarray(cropList[0])),
                                                            cropList, name)
            controller.show_frame(DataEnrollmentPage)
        else:
            crop = videoCapture()
            list(controller.frames.values())[n].update_data(cf, ImageTk.PhotoImage(image=Image.fromarray(crop)),
                                                            crop, role)
            controller.show_frame(DataRecognitionPage)
    else:
        list(controller.frames.values())[pages.InformationPage.value - 1].update_data(spoofingMessage)
        controller.show_frame(InformationPage)


# ritorna alla pagina precedente
def back(controller, entryCF, labelError, entryName=None):
    entryCF.delete(0, tk.END)
    entryCF.insert(0, messageCF)
    if entryName is not None:
        entryName.delete(0, tk.END)
        entryName.insert(0, messageN)
    labelError.configure(fg="#f0f0f0")
    if entryName is not None:
        controller.show_frame(StartPage)
    else:
        controller.show_frame(RecognitionChoicePage)


# aggiunge l'utente al sistema
def addUser(photo, cf, name, medicines, delegates):
    # carica i dataset
    gallery_data = np.load("npy_db/gallery_data.npy").tolist()
    gallery_target = np.load("npy_db/gallery_target.npy").tolist()
    gallery_histograms = np.load("npy_db/histogram_gallery_data.npy").tolist()
    medicine_csv = pd.read_csv("dataset_user.csv", index_col=[0])
    # per ogni foto, si genera l'immagine LBP, si genera l'histogram associato e si appende al set predefinito
    for i in range(n_photo_x_user):
        gallery_data.append(photo[i])
        gallery_target.append(cf)
        lbp = Local_Binary_Pattern(radius, neighborhood, photo[i])
        gallery_histograms.append(lbp.createHistogram(lbp.compute_lbp()))
    # si salvano i dataset
    np.save("npy_db/gallery_data.npy", np.array(gallery_data))
    np.save("npy_db/gallery_target.npy", np.array(gallery_target))
    np.save("npy_db/histogram_gallery_data.npy", np.array(gallery_histograms))
    # si aggiornano i thresholds
    updateThreshold(cf)
    print("L'utente", name, "viene aggiunto al dataset")
    print("Il codice fiscale è", cf)
    print(len(gallery_data), len(gallery_target), len(gallery_histograms))
    # si aggiornano i dati dell'utente
    medicine_csv = medicine_csv.append(
        {"User": name, "Codice Fiscale": cf, "Farmaci": medicines, "Delegati": delegates,
         "Data": date.today().strftime("%d/%m/%Y")}, ignore_index=True)
    medicine_csv.to_csv('dataset_user.csv')


# si aggiornano i tresholds
def updateThreshold(new_user):
    gallery_threshold = np.load("npy_db/gallery_thresholds.npy").tolist()
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_histograms = np.load("npy_db/histogram_gallery_data.npy")
    new_index = gallery_target.tolist().index(new_user)
    max = -1
    # ritorna la lista di utenti unici. Si usa il dizionario perche' cosi' l'ordine tra gallery users e gallery
    # thresholds e' lo stesso
    galley_users = list(dict.fromkeys(gallery_target))
    print("AGGIORNAMENTO DEI THRESHOLDS...")
    for user in galley_users:
        # confrontiamo un utente con tutti gli utenti della gallery ma non con se stesso
        if user != new_user:
            index = galley_users.index(user)
            # per ogni template dell'utente
            for i in range(n_photo_x_user):
                thd = Recognition.topMatch(user, gallery_target, gallery_histograms,
                                           gallery_histograms[new_index + i])
                # prendo il threshold massimo e lo arrotondo
                if thd > gallery_threshold[index]:
                    gallery_threshold[index] = thd
                if thd > max:
                    if np.round(thd, 2) <= thd:
                        max = np.round(thd, 2) + 0.01
                    else:
                        max = np.round(thd, 2)
    # l'ultimo max fa riferimento all'ultimo utente appena iscritto, questo proprio perche' si segue l'ordine di
    # gallery_users, che mantiene a sua volta l'ordine di gallery_target
    gallery_threshold.append(max)
    print("IL TUO THRESHOLD:", max, "N. TOTALI DI HISTOGRAM:", len(gallery_threshold))
    np.save("npy_db/gallery_threshold.npy", np.array(gallery_threshold))
    np.save("npy_db/gallery_thresholds.npy", gallery_threshold)
    return


# catture multiple
def multipleCapture():
    list_captures = []
    for i in range(n_photo_x_user):
        crop = videoCapture()
        list_captures.append(crop)
    return list_captures


# funzione che cattura la camera e inquadra l'utente finche' questo non scatta una foto
def videoCapture():
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    crop = None
    while True:
        ret, frame = cap.read()
        dets = detector(frame, 1)
        for i, d in enumerate(dets):
            landmark = predictor(frame, d)
            top = landmark.part(19).y
            left = landmark.part(0).x
            right = landmark.part(16).x
            bottom = landmark.part(8).y
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            crop = frame[top:bottom, left:right]
            try:
                crop = cv2.resize(crop, (64, 64))
            except Exception as e:
                print(str(e))
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return gray


# funzione usata all'esterno di questa classe
def detect_face(img, vis, crop=None):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        landmark = predictor(img, d)
        top = landmark.part(19).y
        left = landmark.part(0).x
        right = landmark.part(16).x
        bottom = landmark.part(8).y
        crop = img[top:bottom, left:right]
        cv2.rectangle(vis, (left, top), (right, bottom), (0, 255, 0), 3)
        try:
            crop = cv2.resize(crop, (dim_image, dim_image))
        except Exception as e:
            print(str(e))
    if len(dets) > 0:
        try:
            cv2.imshow('Face', crop)
        except Exception as e:
            print(str(e))
    return crop


# controlla se il codice fiscale e' correttto
def isCF(cf):
    if len(cf) != 16:
        return False
    return any(i.isdigit() for i in cf)


# main
def main():
    app = Page()
    app.geometry('300x550')
    app.mainloop()


if __name__ == '__main__':
    main()
