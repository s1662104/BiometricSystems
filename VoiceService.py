import speech_recognition as sr
import pyttsx3
import tkinter as tk
import config
import Pages
from difflib import SequenceMatcher
import time

spell = {"A": "Ancona", "B": "Bologna", "C": "Como", "D": "Domodossola", "E": "Empoli", "F": "Firenze", "G": "Genova",
            "H": "Hotel", "I": "Imola", "J": "Jolly", "K": "Cappa", "L": "Livorno", "M": "Milano", "N": "Napoli",
            "O": "Otranto", "P": "Palermo", "Q": "Quarto", "R": "Roma", "S": "Savona", "T": "Torino", "U":"Udine",
            "V": "Venezia", "W": "Washington", "X": "Xilofono", "Y": "Ipsilon", "Z": "Zara"}

numbers = {"uno": 1, "due": 2}

class Voice:

    def __init__(self):
        self.recognizer_instance = sr.Recognizer()  # Crea una istanza del recognizer
        self.synthesis = pyttsx3.init()
        #voices = self.synthesis.getProperty('voices')
        #self.synthesis.setProperty('voice', voices[0].id)
        self.synthesis.setProperty('voice', 'com.apple.speech.synthesis.voice.alice')
        newVoiceRate = 140
        self.synthesis.setProperty('rate', newVoiceRate)
        self.threshold = 0.8

    def speech_recognize(self, higher_pause=False):
        with sr.Microphone() as source:
            self.recognizer_instance.adjust_for_ambient_noise(source)
            if higher_pause:
                self.recognizer_instance.pause_threshold = 3.0
            else:
                self.recognizer_instance.pause_threshold = 2.0
            # TODO: DARE UN FEEDBACK SU QUANDO INIZIARE A PARLARE E QUANDO NON STA PIU' ASCOLTANDO
            print("Sono in ascolto... parla pure!")
            audio = self.recognizer_instance.listen(source)
            print("Ok! sto ora elaborando il messaggio!")
            text = None
            try:
                text = self.recognizer_instance.recognize_google(audio, language="it-IT")
                print("Google ha capito: \n", text)
            except Exception as e:
                self.speech_synthesis(config.errorSpeechRec)
                return self.speech_recognize()
            return text

    def speech_synthesis(self, text):
        self.synthesis.say(text)
        self.synthesis.runAndWait()

    def compare_strings(self, string1, string2):
        return SequenceMatcher(None, string1, string2).ratio() >= self.threshold

class VocalPages:
    def __init__(self, page: Pages.Page):
        self.page = page
        self.voice = Voice()

    def start_page(self, repeat=False):
        if not repeat:
            self.voice.speech_synthesis(config.initialMessage + " " + config.choice1 + " " + config.choice2)
        choice = self.voice.speech_recognize()
        if self.voice.compare_strings(choice,config.choice1.lower()):
            self.voice.speech_synthesis("Operazione scelta "+config.choice1)
            self.page.get_pages()[Pages.StartPage].button1.invoke()
            self.enroll_page_CF()
        elif self.voice.compare_strings(choice,config.choice2.lower()):
            self.voice.speech_synthesis("Operazione scelta " + config.choice2)
            self.page.get_pages()[Pages.StartPage].button2.invoke()
            self.recognition_choice_page()
        else:
            self.voice.speech_synthesis("Scegli tra: "+config.choice1 + " " +config.choice2)
            self.start_page(True)

    def enroll_page_CF(self, first_time=True):
        if first_time:
            self.voice.speech_synthesis(config.messageCF+"\n Ricorda di fare lo spelling e di "
                                                         "dire una parola alla volta")
        else:
            self.voice.speech_synthesis(config.messageCF)
        cf = ""
        while len(cf) < 16:
            text = self.voice.speech_recognize(True)
            cf += self.spelling(text)
        self.voice.speech_synthesis(config.confirmCF)
        self.read_cf(cf)
        self.voice.speech_synthesis(config.confirm)
        if self.confirm():
            self.page.get_pages()[Pages.EnrollmentPage].entryCF.delete(0, tk.END)
            self.page.get_pages()[Pages.EnrollmentPage].entryCF.insert(0, cf)
            self.enroll_page_name()
        else:
            self.enroll_page_CF(False)

    def enroll_page_name(self):
        # si divide tra nome e cognome per riconoscere correttamente i casi di doppi nomi e cognomi composti
        self.voice.speech_synthesis(config.messageN)
        first_name = self.voice.speech_recognize()
        self.voice.speech_synthesis(config.messageC)
        last_name = self.voice.speech_recognize()
        self.voice.speech_synthesis(first_name+" "+last_name +" "+config.confirm)
        if self.confirm():
            self.page.get_pages()[Pages.EnrollmentPage].entryName.delete(0, tk.END)
            self.page.get_pages()[Pages.EnrollmentPage].entryName.insert(0, first_name+" "+last_name)
            self.voice.speech_synthesis(config.messagePhoto)
            self.page.get_pages()[Pages.EnrollmentPage].invio.invoke()
            self.data_enrollment_page()
        else:
            first_name = self.check_name(first_name)
            last_name = self.check_name(last_name)
            self.page.get_pages()[Pages.EnrollmentPage].entryName.delete(0, tk.END)
            self.page.get_pages()[Pages.EnrollmentPage].entryName.insert(0, first_name+" "+last_name)
            self.voice.speech_synthesis(config.messagePhoto)
            self.page.get_pages()[Pages.EnrollmentPage].invio.invoke()
            self.data_enrollment_page()

    # TODO DA SOSTITUIRE
    def data_enrollment_page(self):
        num_medicines = 0
        #self.page.get_pages()[Pages.DataEnrollmentPage].entryNMedicine.delete(0, tk.END)
        #self.page.get_pages()[Pages.DataEnrollmentPage].entryNMedicine.insert(0, 1)
        self.voice.speech_synthesis(config.numberMedicines)
        num_string = self.voice.speech_recognize()

        if numbers.__contains__(num_string):
            num_medicines = numbers.__getitem__(num_string)
        try:
            num_medicines = int(num_string)
        except Exception as e:
            self.voice.speech_synthesis(config.messageError)

        actualN = len(self.page.get_pages()[Pages.DataEnrollmentPage].medicineEntry)
        if num_medicines > actualN:
            n = num_medicines - actualN
            for m in range(n):
                entryMedicine = tk.Entry(self)
                entryMedicine.insert(1, "")
                entryMedicine.pack(pady=1, padx=30)
                self.page.get_pages()[Pages.DataEnrollmentPage].medicineEntry.append(entryMedicine)
        elif num_medicines < actualN:
            n = actualN - num_medicines
            for m in range(n):
                entryMedicine = self.page.get_pages()[Pages.DataEnrollmentPage].medicineEntry[len(self.page.get_pages()[Pages.DataEnrollmentPage].medicineEntry) - 1]
                entryMedicine.destroy()
                self.page.get_pages()[Pages.DataEnrollmentPage].medicineEntry.pop()

        i = 0
        while i < num_medicines:
            self.voice.speech_synthesis(config.messageMedicine)
            entryMedicine = self.voice.speech_recognize()
            self.page.get_pages()[Pages.DataEnrollmentPage].medicineEntry[i].insert(0, entryMedicine)
            print(entryMedicine)
            i += 1
        count = 0
        while count < 1000:
            count += 1
        #self.page.get_pages()[Pages.DataEnrollmentPage].buttonConferma.invoke()
        #self.information_page(config.enrollmentCompleted)


    def information_page(self, info):
        self.voice.speech_synthesis(info)
        count = 0
        while count < 10000000:
            count += 1
        self.page.get_pages()[Pages.InformationPage].homeButton.invoke()
        self.start_page()

    def recognition_choice_page(self):
        self.voice.speech_synthesis(config.messageRecognition+" "+config.recognitionChoice1+" "+
                                    config.recognitionChoice2)
        text = self.voice.speech_recognize()
        if self.voice.compare_strings(text,config.recognitionChoice1.lower()):
            self.voice.speech_synthesis("Ruolo scelto: "+config.recognitionChoice1)
            self.page.get_pages()[Pages.RecognitionChoicePage].button1.invoke()
        elif self.voice.compare_strings(text,config.recognitionChoice2.lower()):
            self.voice.speech_synthesis("Ruolo scelto: " + config.recognitionChoice2)
            self.page.get_pages()[Pages.RecognitionChoicePage].button2.invoke()
        else:
            self.voice.speech_synthesis("Scegli tra: "+config.choice1 + " " +config.choice2)
            self.start_page(True)

    def check_name(self,text,  repeat= False):
        if not repeat:
            self.voice.speech_synthesis(config.errorFirstName + " " + text)
            if self.confirm():
                return text
            else:
                return self.check_name(text,True)
        else:
            self.voice.speech_synthesis(config.errorSpelling)
            nameSpelled = self.voice.speech_recognize(True)
            name = self.spelling(nameSpelled, True)
            print(name)
            self.voice.speech_synthesis(name+" "+config.confirm)
            if self.confirm():
                return name
            else:
                return self.check_name("", True)

    def spelling(self, text, name=False):
        result = ""
        words = text.split(" ")
        for w in words:
            if w.__contains__("-"):
                result += w.split("-")[0][0].upper()
                result += w.split("-")[1][0].upper()
            else:
                if w.isdigit():
                    result += w
                else:
                    result += w.upper()[0]
        if name:
            result = result.capitalize()
        print(result)
        return result

    def confirm(self):
        text = self.voice.speech_recognize()
        print(text)
        return self.voice.compare_strings(text, config.yes.lower())

    def read_cf(self, cf):
        for c in cf:
            if c.isalpha():
                self.voice.speech_synthesis(spell[c])
            else:
                self.voice.speech_synthesis(c)

if __name__ == '__main__':
    voice = Voice()
    #voice.speech_synthesis(config.initialMessage + " " + config.choice1 + " " + config.choice2)
    #choice = voice.speech_recognize()
    #text = config.initialMessage + " " + config.choice1 + " " + config.choice2

    app = Pages.Page()
    app.geometry('300x550')
    #vocal_app = VocalPages(app)
    #task = threading.Thread(target=vocal_app.start_page)
    #task.start()
    app.get_pages()[Pages.StartPage].button1.invoke()
    app.get_pages()[Pages.EnrollmentPage].entryCF.delete(0, tk.END)
    app.get_pages()[Pages.EnrollmentPage].entryCF.insert(0, "WHTCB78P26U522TH")
    app.get_pages()[Pages.EnrollmentPage].entryName.delete(0, tk.END)
    app.get_pages()[Pages.EnrollmentPage].entryName.insert(0, "Alessandro Mazzucchelli")
    app.get_pages()[Pages.EnrollmentPage].invio.invoke()

    num_medicines = 0
    voice.speech_synthesis(config.numberMedicines)
    num_string = voice.speech_recognize()

    if numbers.__contains__(num_string):
        num_medicines = numbers.__getitem__(num_string)
    try:
        num_medicines = int(num_string)
    except Exception as e:
        voice.speech_synthesis(config.messageError)

    actualN = len(app.get_pages()[Pages.DataEnrollmentPage].medicineEntry)
    if num_medicines > actualN:
        n = num_medicines - actualN
        for m in range(n):
            entryMedicine = tk.Entry()
            entryMedicine.insert(1, "")
            entryMedicine.pack(pady=1, padx=30)
            app.get_pages()[Pages.DataEnrollmentPage].medicineEntry.append(entryMedicine)
    elif num_medicines < actualN:
        n = actualN - num_medicines
        for m in range(n):
            entryMedicine = app.get_pages()[Pages.DataEnrollmentPage].medicineEntry[len(app.get_pages()[Pages.DataEnrollmentPage].medicineEntry) - 1]
            entryMedicine.destroy()
            app.get_pages()[Pages.DataEnrollmentPage].medicineEntry.pop()

    i = 0
    while i < num_medicines:
        voice.speech_synthesis(config.messageMedicine)
        entryMedicine = voice.speech_recognize()
        app.get_pages()[Pages.DataEnrollmentPage].medicineEntry[i].insert(0, entryMedicine)
        print(entryMedicine)
        i += 1
    count = 0
    while count < 1000:
        count += 1
    #app.get_pages()[Pages.DataEnrollmentPage].buttonConferma.invoke()
    app.mainloop()
