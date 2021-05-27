import dlib
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

dim_image = 64
number_maximum_delegate = 3
n_photo_x_user = 5

class Page(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        self.frames = {}

        for F in (StartPage, EnrollmentPage, RecognitionPage, DataEnrollmentPage, DataRecognitionPage,
                  InformationPage, UserPage, RecognitionChoicePage):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


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
            list(controller.frames.values())[1].reset()
            controller.show_frame(EnrollmentPage)

        def goToRecognize():
            controller.show_frame(RecognitionChoicePage)


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

        # tk.Button(self, text="Indietro", width=8, height=1, bg='#1E79FA',
        #           command=lambda: back(controller, self.entryCF, labelError, self.entryName)).place(y=400,x=2)

        tk.Button(self, text="Indietro", width=8, height=1, bg='#1E79FA',
                  command=lambda: back(controller, self.entryCF, labelError, self.entryName)).pack(side="left",
                                                                                                   pady=300)

    def reset(self):
        self.entryCF.delete(0, tk.END)
        self.entryCF.insert(0, messageCF)
        self.entryName.delete(0, tk.END)
        self.entryName.insert(0, messageN)


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

        def confirm(role):
            list(controller.frames.values())[2].reset()
            list(controller.frames.values())[2].update_data(role)
            controller.show_frame(RecognitionPage)


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

    def reset(self):
        self.entryCF.delete(0, tk.END)
        self.entryCF.insert(0, messageCF)

    def update_data(self,role):
        self.role = role


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

    def reset(self):
        self.entryNMedicine.delete(0, tk.END)
        self.entryNMedicine.insert(0, "")
        self.addMedicines("0")
        for i in range(number_maximum_delegate):
            self.delegateEntry[i].delete(0, tk.END)
            self.delegateEntry[i].insert(0, "")

    def update_data(self, cf, img, photo, name):
        print(name)
        self.name.config(text="NOME: " + name)
        self.cf.config(text="CODICE FISCALE: " + cf)
        self.panel.config(image=img)
        self.panel.image = img
        print(img)
        self.photo = photo

    def addMedicines(self, nMedicine):
        if not nMedicine.isdigit():
            self.bell()
        else:
            n = int(nMedicine)
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

    def back(self, controller):
        controller.show_frame(EnrollmentPage)

    def confirm(self, controller):
        medicineError = False
        if len(self.medicineEntry) == 0:
            medicineError = True
        else:
            for medicine in self.medicineEntry:
                if medicine.get() == "":
                    medicineError = True
                    break
        if medicineError:
            messagebox.showerror(title="Errore", message="Inserisci i farmaci")
        else:
            delegatesError = False
            delegates = []
            for delegate in self.delegateEntry:
                if delegate.get() != "":
                    if not isCF(delegate.get()):
                        delegatesError = True
                        break
                    else:
                        delegates.append(delegate.get())
            if delegatesError:
                messagebox.showerror(title="Errore", message="Inserisci i codici fiscali dei tuoi delegati")
            else:
                medicines = []
                for medicine in self.medicineEntry:
                    medicines.append(medicine.get())
                addUser(self.photo, self.cf.cget("text")[16:], self.name.cget("text")[6:], medicines, delegates)
                list(controller.frames.values())[5].update_data(enrollmentCompleted)
                controller.show_frame(InformationPage)


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

    def reset(self):
        pass

    def update_data(self, cf, img, photo, role, name=None):
        self.role = role
        self.cf.config(text="CODICE FISCALE: " + cf)
        self.panel.config(image=img)
        self.panel.image = img
        self.photo = photo

    def confirm(self, controller):
        print("role", self.role)
        if self.role == 0:
            print("REC")
            user, index, patient = Recognition.recognize(self.cf.cget("text")[16:], self.photo)
        else:
            print("IDE")
            user, index, patient = Recognition.identify(self.cf.cget("text")[16:], self.photo)
        print(user)
        if user is not None:
            list(controller.frames.values())[6].reset()
            list(controller.frames.values())[6].update_data(index, user["User"], patient["User"], user["Codice Fiscale"],
                                                            user["Delegati"], user["Farmaci"], user["Data"],
                                                            self.panel.image)
            controller.show_frame(UserPage)
        else:
            list(controller.frames.values())[5].update_data(recognitionRejected)
            controller.show_frame(InformationPage)

    def back(self, controller):
        controller.show_frame(RecognitionPage)


class InformationPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.label = tk.Label(self, text="")
        self.label.pack(pady=200)

        tk.Button(self, text="Home", width=8, height=1, bg='#1E79FA',
                  command=lambda: controller.show_frame(StartPage)).place(y=520, x=110)

    def update_data(self, info):
        self.label.config(text=info)


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

    def reset(self):
        for i, label in enumerate(self.delegatesLabels):
            self.delegatesLabels[i].config(text="-")
        for entry in self.entries:
            entry.destroy()

    def update_data(self, index, name, patient, cf, delegates, medicines, last_date, photo):
        self.cf.config(text="CODICE FISCALE: " + cf)
        self.name.config(text="UTENTE RICONOSCIUTO: " + patient)
        self.patient.config(text="PAZIENTE: " + name)
        self.panel.config(image=photo)
        self.panel.image = photo
        delegates = ast.literal_eval(delegates)
        for i, label in enumerate(self.delegatesLabels):
            if i < len(delegates):
                self.delegatesLabels[i].config(text=delegates[i])
            else:
                self.delegatesLabels[i].config(text="-")
        medicines = ast.literal_eval(medicines)
        for medicine in medicines:
            label = tk.Label(self, text=medicine)
            label.pack()
            self.entries.append(label)
        label = tk.Label(self, text="FARMACI PRELEVATI:")
        label.pack()
        self.entries.append(label)
        self.obtainable_medicines(last_date, medicines)
        csv = pd.read_csv("dataset_user.csv", index_col=[0])
        csv.iloc[index]["Data"] = date.today().strftime("%d/%m/%Y")
        csv.to_csv('dataset_user.csv')

    def obtainable_medicines(self, d: str, medicines):
        dmy = d.split("/")
        last_date = date(int(dmy[2]), int(dmy[1]), int(dmy[0]))
        today = date.today()
        days = (today - last_date).days
        medicine_dataset = pd.read_csv("dataset_medicine.csv")
        for row in medicine_dataset.iterrows():
            for medicine in medicines:
                s = medicine.split(" ")
                name = ""
                for i, val in enumerate(s):
                    if i < len(s) - 2:
                        name += s[i]
                dose = s[len(s) - 2] + " " + s[len(s) - 1]
                if row[1]["Nome"] == name and row[1]["Dosaggio"] == dose:
                    n = row[1]["Numero Pasticche"]
                    dose_x_day = row[1]["Dose x giorno"]
                    box = int(days / n) * dose_x_day
                    label = tk.Label(self, text=medicine + " x " + str(box))
                    label.pack()
                    self.entries.append(label)


def check_input(controller, cf, labelError, op, role=None, name=None):
    if len(cf) != 16 or cf == messageCF or (name is not None and name == messageN):
        labelError.configure(fg="red")
        return
    else:
        labelError.configure(fg="#f0f0f0")
    # crop = videoCapture()
    if op == 0:
        n = 3
    else:
        n = 4
    list(controller.frames.values())[n].reset()
    if op == 0:
        # Inizio parte antispoofing in fase di registrazione
        user = False
        nameFileCsv = 'histogram.csv'
        if (EyeBlink(None).eyeBlinkStart()) == False:
            user = False
        elif (MicroTexture(nameFileCsv).microTextureCam() == False):
            print("REPLAY ATTACK E' FALSE")
            user = False
        else:
            user = True
        # Fine parte antispoofing in fase di registrazione
        if (user == True):
            print("USER == TRUE")
            cropList = multipleCapture()

        # list(controller.frames.values())[n].update_data(cf, ImageTk.PhotoImage(image=Image.fromarray(crop)), crop,
        #                                                name)
            list(controller.frames.values())[n].update_data(cf, ImageTk.PhotoImage(image=Image.fromarray(cropList[0])),
                                                        cropList, name)
            controller.show_frame(DataEnrollmentPage)
        else:
            list(controller.frames.values())[5].update_data(spoofingMessage)
            controller.show_frame(InformationPage)

    else:
        crop = videoCapture()
        ##Inizio parte antispoofing in fase di matching
        user = False
        nameFileCsv = 'histogram.csv'
        if (EyeBlink(None).eyeBlinkStart()) == False:
            user = False
        elif (MicroTexture(nameFileCsv).microTextureCam() == False):
            print("REPLAY ATTACK E' FALSE")
            user = False
        else:
            user = True
        ##Fine parte antispoofing in fase di matching
        print("E' terminato l'anti spoofing")
        if (user == True):
            print("USER == TRUE")
            list(controller.frames.values())[n].update_data(cf, ImageTk.PhotoImage(image=Image.fromarray(crop)),
                                                            crop, role)
            controller.show_frame(DataRecognitionPage)
        else:
            list(controller.frames.values())[5].update_data(spoofingMessage)
            controller.show_frame(InformationPage)

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


def addUser(photo, cf, name, medicines, delegates):
    gallery_data = np.load("npy_db/gallery_data.npy").tolist()
    gallery_target = np.load("npy_db/gallery_target.npy").tolist()
    gallery_histograms = np.load("npy_db/histogram_gallery_data.npy").tolist()
    medicine_csv = pd.read_csv("dataset_user.csv", index_col=[0])
    for i in range(n_photo_x_user):
        gallery_data.append(photo[i])
        gallery_target.append(cf)
        lbp = Local_Binary_Pattern(1,8,photo[i])
        gallery_histograms.append(lbp.createHistogram(lbp.compute_lbp()))
    np.save("npy_db/gallery_data.npy", np.array(gallery_data))
    np.save("npy_db/gallery_target.npy", np.array(gallery_target))
    np.save("npy_db/histogram_gallery_data.npy", np.array(gallery_histograms))
    updateThreshold(cf)
    print("L'utente", name, "viene aggiunto al dataset")
    print("Il codice fiscale è", cf)
    print(len(gallery_data), len(gallery_target),len(gallery_histograms))
    # print(photo, gallery_data[len(gallery_data) - 1])
    # print(cf, gallery_target[len(gallery_target) - 1])
    medicine_csv = medicine_csv.append(
        {"User": name, "Codice Fiscale": cf, "Farmaci": medicines, "Delegati": delegates,
         "Data": date.today().strftime("%d/%m/%Y")}, ignore_index=True)
    medicine_csv.to_csv('dataset_user.csv')

#def updateThreshold(new_user):
#    gallery_target = np.load("npy_db/gallery_target.npy")
#    gallery_threshold = np.load("npy_db/gallery_thresholds.npy").tolist()
#    histogram_gallery_data = np.load("npy_db/histogram_gallery_data.npy")
#    new_index = gallery_target.index(new_user)
#    max_thd = -1
#    for user in np.unique(gallery_target):
#        if user != new_user:
#            index = gallery_target.index(user)
#            for i in range(5):
#                thd = Recognition.topMatch(user, gallery_target, histogram_gallery_data, histogram_gallery_data[new_index+i])
#                val = Recognition.topMatch(new_user, gallery_target, histogram_gallery_data,histogram_gallery_data[index+i])
#                if thd > gallery_threshold[index]:
#                    gallery_threshold[index] = np.round(thd,2)
#                if val > max_thd:
#                    max_thd = np.round(val,2)
#    gallery_threshold.append(max_thd)
#    np.save("npy_db/gallery_thresholds.npy", gallery_threshold)
#    return

def updateThreshold(new_user):
    gallery_threshold = np.load("npy_db/gallery_thresholds.npy").tolist()
    gallery_target = np.load("npy_db/gallery_target.npy")
    gallery_histograms = np.load("npy_db/histogram_gallery_data.npy")
    new_index = gallery_target.tolist().index(new_user)
    max = -1
    galley_users = list(dict.fromkeys(gallery_target))
    for user in galley_users:
        if user != new_user:
            index = galley_users.index(user)
            for i in range(5):
                thd = Recognition.topMatch(user, gallery_target, gallery_histograms,
                                           gallery_histograms[new_index + i])

                if thd > gallery_threshold[index]:
                    gallery_threshold[index] = thd
                if thd > max:
                    max = thd + 0.00000000001
    gallery_threshold.append(max)
    print("IL TUO THRESHOLD:",max, "N. TOTALI DI HISTOGRAM:",len(gallery_threshold))
    np.save("npy_db/gallery_threshold.npy", np.array(gallery_threshold))
    np.save("npy_db/gallery_thresholds.npy", gallery_threshold)
    return

def multipleCapture():
    list_captures = []
    for i in range(n_photo_x_user):
        crop = videoCapture()
        list_captures.append(crop)
    return list_captures

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


# def videoCapture():
#     cap = cv2.VideoCapture(0)
#     # while (True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     vis = frame.copy()
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     crop = None
#     while crop is None:
#         print("CROP NON FATTO!")
#         crop = detect_face(gray, vis)

# # # Display the resulting frame
# # cv2.imshow('frame', vis)
# # if cv2.waitKey(1) & 0xFF == ord('q'):
# #     break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
# return crop


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


def isCF(cf):
    if len(cf) != 16:
        return False
    return any(i.isdigit() for i in cf)


def main():
    app = Page()
    app.geometry('300x550')
    app.mainloop()


if __name__ == '__main__':
    main()
