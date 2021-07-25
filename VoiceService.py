import speech_recognition as sr
import pyttsx3
import tkinter as tk
import config
import Pages
from difflib import SequenceMatcher

spell = {"A": "Ancona", "B": "Bologna", "C": "Como", "D": "Domodossola", "E": "Empoli", "F": "Firenze", "G": "Genova",
            "H": "Hotel", "I": "Imola", "J": "Jolly", "K": "Cappa", "L": "Livorno", "M": "Milano", "N": "Napoli",
            "O": "Otranto", "P": "Palermo", "Q": "Quarto", "R": "Roma", "S": "Savona", "T": "Torino", "U":"Udine",
            "V": "Venezia", "W": "Washington", "X": "Xilofono", "Y": "Ipsilon", "Z": "Zara"}

class Voice:

    def __init__(self):
        self.recognizer_instance = sr.Recognizer()  # Crea una istanza del recognizer
        self.synthesis = pyttsx3.init()
        voices = self.synthesis.getProperty('voices')
        self.synthesis.setProperty('voice', voices[0].id)
        newVoiceRate = 130
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
            self.voice.speech_synthesis(config.initialMessage + " " + config.choice1 + " " +
                                    config.choice2)
        choice = self.voice.speech_recognize()
        if self.voice.compare_strings(choice,config.choice1.lower()):
            print("L'UTENTE HA SCELTO:", config.choice1)
            self.voice.speech_synthesis("Operazione scelta "+config.choice1)
            self.page.get_pages()[Pages.StartPage].button1.invoke()
            self.enroll_page_CF()
        elif self.voice.compare_strings(choice,config.choice1.lower()):
            print("L'UTENTE HA SCELTO:", config.choice1)
            self.page.get_pages()[Pages.StartPage].button2.invoke()
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
        self.page.get_pages()[Pages.DataEnrollmentPage].entryNMedicine.delete(0, tk.END)
        self.page.get_pages()[Pages.DataEnrollmentPage].entryNMedicine.insert(0, 1)
        self.page.get_pages()[Pages.DataEnrollmentPage].buttonInvia.invoke()
        self.page.get_pages()[Pages.DataEnrollmentPage].medicineEntry[0].insert(0, "Prefolic 15 mg")
        count = 0
        while count < 1000:
            count += 1
        self.page.get_pages()[Pages.DataEnrollmentPage].buttonConferma.invoke()
        self.information_page(config.enrollmentCompleted)

    def information_page(self, info):
        self.voice.speech_synthesis(info)
        count = 0
        while count < 10000000:
            count += 1
        self.page.get_pages()[Pages.InformationPage].homeButton.invoke()
        self.start_page()

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
    voice.speech_recognize()
    # voice.speech_synthesis("Prova?")
