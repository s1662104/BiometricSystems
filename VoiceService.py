import speech_recognition as sr
import pyttsx3
import tkinter as tk
import config
import Pages

class Voice:
    def __init__(self):
        self.recognizer_instance = sr.Recognizer()  # Crea una istanza del recognizer
        self.synthesis = pyttsx3.init()
        voices = self.synthesis.getProperty('voices')
        self.synthesis.setProperty('voice', voices[0].id)
        newVoiceRate = 120
        self.synthesis.setProperty('rate', newVoiceRate)

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


class VocalPages:
    def __init__(self, page: Pages.Page):
        self.page = page
        self.voice = Voice()

    def startPage(self):
        self.voice.speech_synthesis(config.messageWelcome.replace("desideri","desidéri"))
        self.voice.speech_synthesis(config.choice1)
        self.voice.speech_synthesis(config.choice2)
        choice = self.voice.speech_recognize()
        if choice.__contains__(config.choice1.lower()):
            print("L'UTENTE HA SCELTO:",config.choice1)
            self.page.get_pages()[Pages.StartPage].button1.invoke()
        elif choice.__contains__(config.choice2):
            print("L'UTENTE HA SCELTO:",config.choice1)
            self.page.get_pages()[Pages.StartPage].button2.invoke()
        else:
            print("ERRORE")


if __name__ == '__main__':
    voice = Voice()
    voice.speech_recognize()
    voice.speech_synthesis("Prova?")
