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

messageBenvenuto = "Benvenuto! \nCosa vuoi fare? \n0. Registrazione \n1. Riconoscimento"
messageA = "Inserire scelta: "
messageCF = "Inserire codice fiscale: "
messageError = "Input non valido"
messageN = "Inserire nome: "
choice1 = "Registrazione"
choice2 = "Riconoscimento"
messageWelcome = "Benvenuto\n Che operazione desideri svolgere?"
numberMedicines = "Quanti farmaci assumi?"
numberDelegate = "A chi vuoi delegare il prelievo di farmaci?"

dim_image = 64
number_maximum_delegate = 3


class Page(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        self.frames = {}

        for F in (StartPage, EnrollmentPage, RecognitionPage, DataEnrollmentPage, DataRecognitionPage,
                  EnrollmentCompleted, UserPage):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(DataRecognitionPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=messageWelcome)
        label.pack(pady=10, padx=50)
        button1 = tk.Button(self, text=choice1, width=15, height=2, bg='#1E79FA',
                            command=lambda: controller.show_frame(EnrollmentPage))
        button2 = tk.Button(self, text=choice2, width=15, height=2, bg='#1E79FA',
                            command=lambda: controller.show_frame(RecognitionPage))
        button1.pack()
        button2.pack(pady=1)


class EnrollmentPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=choice1)
        label.pack(pady=10, padx=10)

        entryCF = tk.Entry(self)
        entryCF.insert(1, messageCF)
        entryCF.pack(pady=2)
        entryName = tk.Entry(self)
        entryName.insert(1, messageN)
        entryName.pack(pady=2)
        button2 = tk.Button(self, text="Invia", width=10, height=1, bg='#1E79FA',
                            command=lambda: check_input(controller, entryCF.get(), labelError, 0, entryName.get()))
        button2.pack()

        labelError = tk.Label(self, text=messageError, fg="#f0f0f0")
        labelError.pack(pady=10, padx=10)

        tk.Button(self, text="Indietro", width=8, height=1, bg='#1E79FA',
                  command=lambda: back(controller, entryCF, labelError, entryName)).pack(side="bottom", pady=240,
                                                                                         padx=2)


class RecognitionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=choice2)
        label.pack(pady=10, padx=10)

        entryCF = tk.Entry(self)
        entryCF.insert(1, messageCF)
        entryCF.pack(padx=0, pady=0)
        button2 = tk.Button(self, text="Invia", width=10, height=1, bg='#1E79FA',
                            command=lambda: check_input(controller, entryCF.get(), labelError, 1))
        button2.pack()

        labelError = tk.Label(self, text=messageError, fg="#f0f0f0")
        labelError.pack(pady=10, padx=10)

        tk.Button(self, text="Indietro", width=8, height=1, bg='#1E79FA',
                  command=lambda: back(controller, entryCF, labelError)).place(y=520, x=2)


class DataPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        photo = np.ones((64, 64)) * 150
        img = ImageTk.PhotoImage(image=Image.fromarray(photo))
        self.panel = tk.Label(self, image=img)
        self.panel.image = img
        self.panel.pack(pady=10, padx=10)

        self.name = tk.Label(self, text="NOME")
        self.name.pack()

        self.cf = tk.Label(self, text="CF")
        self.cf.pack()


class DataEnrollmentPage(DataPage):
    def __init__(self, parent, controller):
        DataPage.__init__(self, parent, controller)

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
        entryNMedicine = tk.Entry(self)
        entryNMedicine.insert(1, "")
        entryNMedicine.pack(padx=0, pady=0)
        button = tk.Button(self, text="Invia", width=10, height=1, bg='#1E79FA',
                           command=lambda: self.addMedicines(entryNMedicine.get()))
        button.pack()

        tk.Button(self, text="Confirma", width=8, height=1, bg='#1E79FA',
                  command=lambda: self.confirm()).place(y=520, x=220)

        tk.Button(self, text="Indietro", width=8, height=1, bg='#1E79FA',
                  command=lambda: self.back(controller)).place(y=520, x=2)

    def update_data(self, cf, img, name=None):
        self.name.config(text="NOME: " + name)
        self.cf.config(text="CODICE FISCALE: " + cf)
        self.panel.config(image=img)
        self.panel.image = img
        self.photo = img

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

    def confirm(self):
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
                print(delegates)
                addUser(self.photo, self.cf.cget("text"), self.name.cget("text"), medicines, delegates)


class DataRecognitionPage(DataPage):
    def __init__(self, parent, controller):
        DataPage.__init__(self, parent, controller)

        self.name.destroy()

        tk.Button(self, text="Confirma", width=8, height=1, bg='#1E79FA',
                  command=lambda: self.confirm(controller)).place(y=520, x=220)

        tk.Button(self, text="Indietro", width=8, height=1, bg='#1E79FA',
                  command=lambda: self.back(controller)).place(y=520, x=2)

    def update_data(self, cf, img, name=None):
        self.cf.config(text="CODICE FISCALE: " + cf)
        self.panel.config(image=img)
        self.panel.image = img
        # self.photo = img

    def confirm(self, controller):
        user,index = Recognition.recognize()
        print(user)
        list(controller.frames.values())[6].update_data(index, user["User"], user["Codice Fiscale"],
                                                        user["Delegati"], user["Farmaci"], user["Data"], self.panel.image)
        controller.show_frame(UserPage)

    def back(self, controller):
        controller.show_frame(RecognitionPage)


class EnrollmentCompleted(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="REGISTRAZIONE COMPLETATA!")
        label.pack(pady=200)

        tk.Button(self, text="Home", width=8, height=1, bg='#1E79FA',
                  command=lambda: controller.show_frame(StartPage)).place(y=520, x=110)


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

    def update_data(self, index, name, cf, delegates, medicines, last_date, photo):
        self.cf.config(text="CODICE FISCALE: " + cf)
        self.name.config(text="NOME: " + name)
        print(photo)
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
        tk.Label(self, text="FARMACI PRELEVATI:").pack()
        self.obtainable_medicines(last_date, medicines)
        csv = pd.read_csv("dataset_user.csv", index_col=[0])
        csv.iloc[index]["Data"] = date.today().strftime("%d/%m/%Y")
        csv.to_csv('dataset_user.csv')

    def obtainable_medicines(self,d: str, medicines):
        dmy = d.split("/")
        last_date = date(int(dmy[2]),int(dmy[1]),int(dmy[0]))
        today = date.today()
        days = (today - last_date).days
        print(days)
        medicine_dataset = pd.read_csv("dataset_medicine.csv")
        for row in medicine_dataset.iterrows():
            for medicine in medicines:
                s = medicine.split(" ")
                name = ""
                print(s)
                for i, val in enumerate(s):
                    if i < len(s)-2:
                        name += s[i]
                print(name)
                dose = s[len(s)-2] + " " + s[len(s)-1]
                print(dose)
                if row[1]["Nome"] == name and row[1]["Dosaggio"] == dose:
                    n = row[1]["Numero Pasticche"]
                    dose_x_day = row[1]["Dose x giorno"]
                    box = int(days / n) * dose_x_day
                    tk.Label(self, text=medicine+" x "+str(box)).pack()


def check_input(controller, cf, labelError, op, name=None):
    if len(cf) != 16 or cf == messageCF or (name is not None and name == messageN):
        labelError.configure(fg="red")
        return
    else:
        labelError.configure(fg="#f0f0f0")
    crop = videoCapture()
    if op == 0:
        n = 3
    else:
        n = 4
    list(controller.frames.values())[n].update_data(cf, ImageTk.PhotoImage(image=Image.fromarray(crop)), name)
    if op == 0:
        controller.show_frame(DataEnrollmentPage)
    else:
        controller.show_frame(DataRecognitionPage)


def back(controller, entryCF, labelError, entryName=None):
    entryCF.delete(0, tk.END)
    entryCF.insert(0, messageCF)
    if entryName != None:
        entryName.delete(0, tk.END)
        entryName.insert(0, messageN)
    labelError.configure(fg="#f0f0f0")
    controller.show_frame(StartPage)


def addUser(photo, cf, name, medicines, delegates):
    gallery_data = np.load("npy_db/gallery_data.npy").tolist()
    gallery_target = np.load("npy_db/gallery_target.npy").tolist()
    medicine_csv = pd.read_csv("dataset_user.csv", index_col=[0])
    gallery_data.append(photo)
    gallery_target.append(cf)
    print(len(gallery_data), len(gallery_target))
    print(photo, gallery_data[len(gallery_data) - 1])
    print(cf, gallery_target[len(gallery_target) - 1])
    medicine_csv = medicine_csv.append(
        {"User": name, "Codice Fiscale": cf, "Farmaci": medicines, "Delegati": delegates}, ignore_index=True)
    np.save("npy_db/gallery_data.npy", gallery_data)
    np.save("npy_db/gallery_target.npy", gallery_target)
    medicine_csv.to_csv('dataset_user.csv')

def videoCapture():
    cap = cv2.VideoCapture(0)
    # while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    vis = frame.copy()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = detect_face(gray, vis)

    # # Display the resulting frame
    # cv2.imshow('frame', vis)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return crop


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
    app.geometry('290x550')
    app.mainloop()


if __name__ == '__main__':
    main()
