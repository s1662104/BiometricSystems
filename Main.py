import dlib
import cv2
import Recognition
import Enrollment
import tkinter as tk

messageBenvenuto = "Benvenuto! \nCosa vuoi fare? \n0. Registrazione \n1. Riconoscimento"
messageA = "Inserire scelta: "
messageCF = "Inserire codice fiscale: "
messageError = "Input non valido"
messageN = "Inserire nome: "
choice1 = "Registrazione"
choice2 = "Riconoscimento"
messageWelcome = "Benvenuto\n Che operazione desideri svolgere?"

dim_image = 64


class Page(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        self.frames = {}

        for F in (StartPage, EnrollmentPage, RecognitionPage):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text=messageWelcome)
        label.pack(pady=10,padx=50)
        button1 = tk.Button(self, text=choice1, width=15, height=2, bg='#1E79FA',
                            command=lambda: controller.show_frame(EnrollmentPage))
        button2 = tk.Button(self, text=choice2, width=15, height=2, bg='#1E79FA',
                            command=lambda: controller.show_frame(RecognitionPage))
        button1.pack()
        button2.pack(pady=10)

class EnrollmentPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=choice1)
        label.pack(pady=10,padx=10)

        entryCF = tk.Entry(self)
        entryCF.insert(1, messageCF)
        entryCF.pack(pady=2)
        entryName = tk.Entry(self)
        entryName.insert(1, messageN)
        entryName.pack(pady=2)
        button2 = tk.Button(self, text="Invia", width=10, height=1, bg='#1E79FA',
                            command=lambda: checkInput(entryCF.get(), labelError, entryName.get()))
        button2.pack()

        labelError = tk.Label(self, text=messageError,fg="#f0f0f0")
        labelError.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="Back", width=15, height=2, bg='#1E79FA',
                            command=lambda: back(controller, entryCF, labelError, entryName))
        button1.pack(pady=100)

class RecognitionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=choice2)
        label.pack(pady=10,padx=10)

        entryCF = tk.Entry(self)
        entryCF.insert(1, messageCF)
        entryCF.pack(padx=0, pady=0)
        button2 = tk.Button(self, text="Invia", width=10, height=1, bg='#1E79FA',
                            command=lambda: checkInput(entryCF.get(), labelError))
        button2.pack()

        labelError = tk.Label(self, text=messageError, fg="#f0f0f0")
        labelError.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="Back", width=15, height=2, bg='#1E79FA',
                            command=lambda: back(controller, entryCF, labelError))
        button1.pack(pady=125)

def checkInput(cf, labelError, name=None):
    if len(cf) != 16 or cf == messageCF or (name!=None and name == messageN):
        labelError.configure(fg="red")
        return
    else:
        labelError.configure(fg="#f0f0f0")
    videoCapture()

def back(controller, entryCF, labelError, entryName=None):
    entryCF.delete(0,tk.END)
    entryCF.insert(0,messageCF)
    if entryName!= None:
        entryName.delete(0, tk.END)
        entryName.insert(0, messageN)
    labelError.configure(fg="#f0f0f0")
    controller.show_frame(StartPage)

def videoCapture():
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        vis = frame.copy()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        crop = detect_face(gray, vis)

        # Display the resulting frame
        cv2.imshow('frame', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
        crop = cv2.resize(crop, (dim_image, dim_image))
    if len(dets) > 0:
        cv2.imshow('Face', crop)
    return crop

def main():
    app = Page()
    app.geometry('300x300')
    app.mainloop()

if __name__ == '__main__':
    main()