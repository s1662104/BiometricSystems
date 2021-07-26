import speech_recognition as sr
import pyttsx3
import tkinter as tk
import config
import Pages
from difflib import SequenceMatcher


class Voice:

    def __init__(self):
        self.recognizer_instance = sr.Recognizer()  # Crea una istanza del recognizer
        self.synthesis = pyttsx3.init()
        voices = self.synthesis.getProperty('voices')
        self.synthesis.setProperty('voice', voices[0].id)
        newVoiceRate = 120
        self.synthesis.setProperty('rate', newVoiceRate)
        self.threshold = 0.8

    def speech_recognize(self):
        with sr.Microphone() as source:
            self.recognizer_instance.adjust_for_ambient_noise(source)
            # TODO: DARE UN FEEDBACK SU QUANDO INIZIARE A PARLARE E QUANDO NON STA PIU' ASCOLTANDO
            print("Sono in ascolto... parla pure!")
            audio = self.recognizer_instance.listen(source)
            print("Ok! sto ora elaborando il messaggio!")
            text = None
        try:
            text = self.recognizer_instance.recognize_google(audio, language="it-IT")
            print("Google ha capito: \n", text)
        except Exception as e:
            print(e)
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

    def start_page(self):
        self.voice.speech_synthesis(config.messageWelcome.replace("desideri", "desidéri") + " " + config.choice1 + " " +
                                    config.choice2)
        choice = self.voice.speech_recognize()
        if self.voice.compare_strings(choice,config.choice1.lower()):
            print("L'UTENTE HA SCELTO:", config.choice1)
            self.page.get_pages()[Pages.StartPage].button1.invoke()
            self.enroll_page()
        elif self.voice.compare_strings(choice,config.choice1.lower()):
            print("L'UTENTE HA SCELTO:", config.choice1)
            self.page.get_pages()[Pages.StartPage].button2.invoke()
        else:
            print("ERRORE")

    def enroll_page(self):
        self.voice.speech_synthesis(config.messageCF+"\n Ricorda di fare lo spelling e di dire una parola alla volta")
        count = 0
        cf = ""
        while count < 5:
            text = self.voice.speech_recognize()
            words = text.split(" ")
            for w in words:
                if w.isdigit():
                    cf += w
                    count += len(w)
                else:
                    cf += w.upper()[0]
                    count += 1
        self.page.get_pages()[Pages.EnrollmentPage].entryCF.delete(0, tk.END)
        self.page.get_pages()[Pages.EnrollmentPage].entryCF.insert(0, cf)

    def data_enrollment_page(self):
        self.voice.speech_synthesis(config.numberDelegate)

if __name__ == '__main__':
    # voice = Voice()
    # voice.speech_recognize()
    # voice.speech_synthesis("Prova?")
    vp = VocalPages(None)
    vp.enroll_page()
