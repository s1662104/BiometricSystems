import speech_recognition as sr
import pyttsx3
import tkinter as tk
import config
import Pages
from difflib import SequenceMatcher
import pandas as pd
import Levenshtein

spell = {"A": "Ancona", "B": "Bologna", "C": "Como", "D": "Domodossola", "E": "Empoli", "F": "Firenze", "G": "Genova",
            "H": "Hotel", "I": "Imola", "J": "Jolly", "K": "Cappa", "L": "Livorno", "M": "Milano", "N": "Napoli",
            "O": "Otranto", "P": "Palermo", "Q": "Quarto", "R": "Roma", "S": "Savona", "T": "Torino", "U":"Udine",
            "V": "Venezia", "W": "Washington", "X": "Xilofono", "Y": "Ipsilon", "Z": "Zara"}

numbers = {"uno": 1, "due": 2}

class Voice:

    def __init__(self):
        self.recognizer_instance = sr.Recognizer()  # Crea una istanza del recognizer
        self.synthesis = pyttsx3.init()
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

    def medicine_autocorrect(self, input_word):
        input_word = input_word.lower()
        medicines = pd.read_csv('dataset_medicine.csv', index_col=[0])['Nome']
        medicines = medicines.tolist()
        for i in range(len(medicines)):
            medicines[i] = medicines[i].lower()
        print(medicines)
        if input_word in medicines:
            return input_word
        else:
            min_dist = 100
            word = input_word
            for m in medicines:
                dist = Levenshtein.distance(m, input_word)
                if  dist < min_dist:
                    min_dist = dist
                    word = m
            return word

class VocalPages:
    def __init__(self, page: Pages.Page):
        self.page = page
        self.voice = Voice()

    def start_page(self, repeat=False):
        if not repeat:
            self.voice.speech_synthesis(config.initialMessage + " " + config.choice1 + " " + config.choice2)
            # time.sleep(18)
        choice = self.voice.speech_recognize()
        if self.voice.compare_strings(choice,config.choice1.lower()):
            self.voice.speech_synthesis("Operazione scelta "+config.choice1)
            self.page.current_page.button1.invoke()
            self.enroll_page_CF()
        elif self.voice.compare_strings(choice,config.choice2.lower()):
            self.voice.speech_synthesis("Operazione scelta " + config.choice2)
            self.page.current_page.button2.invoke()
            self.recognition_choice_page()
        else:
            self.voice.speech_synthesis("Scegli tra: "+config.choice1 + " " +config.choice2)
            self.start_page(True)

    def enroll_page_CF(self):
        cf = self.page_CF(False)
        self.page.current_page.entryCF.delete(0, tk.END)
        self.page.current_page.entryCF.insert(0, cf)
        self.enroll_page_name()

    def page_CF(self, first_time=True):
        if first_time:
            self.voice.speech_synthesis(config.messageCF+"\n Ricorda di fare lo spelling e di "
                                                         "dire una parola alla volta")
        else:
            self.voice.speech_synthesis(config.messageCF)
        cf = ""
        # TODO CONTROLLO SULLA LUNGHEZZA DEL CF
        while len(cf) < 16:
            text = self.voice.speech_recognize(True)
            cf += self.spelling(text)
        self.voice.speech_synthesis(config.confirmCF)
        self.read_cf(cf)
        self.voice.speech_synthesis(config.confirm)
        if self.confirm():
            return cf
        else:
            return self.page_CF(False)

    def enroll_page_name(self):
        # si divide tra nome e cognome per riconoscere correttamente i casi di doppi nomi e cognomi composti
        self.voice.speech_synthesis(config.messageN)
        first_name = self.voice.speech_recognize()
        self.voice.speech_synthesis(config.messageC)
        last_name = self.voice.speech_recognize()
        self.voice.speech_synthesis(first_name+" "+last_name +" "+config.confirm)
        if self.confirm():
            self.page.current_page.entryName.delete(0, tk.END)
            self.page.current_page.entryName.insert(0, first_name+" "+last_name)
            self.voice.speech_synthesis(config.messagePhoto)
            self.page.current_page.invio.invoke()
            self.data_enrollment_page()
        else:
            first_name = self.check_name(first_name)
            last_name = self.check_name(last_name)
            self.page.current_page.entryName.delete(0, tk.END)
            self.page.current_page.entryName.insert(0, first_name+" "+last_name)
            self.voice.speech_synthesis(config.messagePhoto)
            self.page.current_page.invio.invoke()
            self.data_enrollment_page()

    def data_enrollment_page(self):
        self.voice.speech_synthesis(config.numberMedicines)
        num_string = self.voice.speech_recognize()

        if numbers.__contains__(num_string):
            num_medicines = numbers.__getitem__(num_string)
        else:
            try:
                num_medicines = int(num_string)
            except Exception as e:
                self.voice.speech_synthesis(config.messageError)
                return self.data_enrollment_page()

        self.voice.speech_synthesis(config.numberMedicinesConfirm + num_medicines + "?")
        if not self.confirm():
            self.data_enrollment_page()
        self.page.current_page.entryNMedicine.delete(0, tk.END)
        self.page.current_page.entryNMedicine.insert(0, num_medicines)
        self.page.current_page.buttonInvia.invoke()

        i = 0
        while i < num_medicines:
            self.voice.speech_synthesis(config.messageMedicine)
            entryMedicine = self.voice.speech_recognize()
            entryMedicine = self.voice.medicine_autocorrect(entryMedicine)
            self.voice.speech_synthesis(config.medicineConfirm+entryMedicine+"?")
            if self.confirm():
                self.page.current_page.medicineEntry[i].insert(0, entryMedicine)
                print(entryMedicine)
                i += 1

        self.voice.speech_synthesis(config.medConfirm)
        if not self.confirm():
            self.voice.speech_synthesis(config.changeMed)
            index = self.voice.speech_recognize()
            

        self.page.current_page.buttonConferma.invoke()
        self.information_page(config.enrollmentCompleted)


    def information_page(self, info):
        self.voice.speech_synthesis(info)
        count = 0
        while count < 10000000:
            count += 1
        self.page.current_page.homeButton.invoke()
        self.start_page()

    def recognition_choice_page(self):
        self.voice.speech_synthesis("Scegli tra "+" "+config.recognitionChoice1+" "+
                                    config.recognitionChoice2)
        text = self.voice.speech_recognize()
        if self.voice.compare_strings(text,config.recognitionChoice1.lower()):
            self.voice.speech_synthesis("Ruolo scelto: "+config.recognitionChoice1)
            self.page.current_page.button1.invoke()
            self.recognition_page()
        elif self.voice.compare_strings(text,config.recognitionChoice2.lower()):
            self.voice.speech_synthesis("Ruolo scelto: " + config.recognitionChoice2)
            self.page.current_page.button2.invoke()
            self.recognition_page()
        else:
            self.recognition_choice_page()

    def recognition_page(self):
        cf = self.page_CF()
        self.page.current_page.entryCF.delete(0, tk.END)
        self.page.current_page.entryCF.insert(0, cf)
        self.voice.speech_synthesis(config.messagePhoto)
        self.page.current_page.buttonInvia.invoke()
        self.data_recognition_page()

    def data_recognition_page(self):
        cf = self.page.current_page.cf.cget("text").split("CODICE FISCALE: ")[1]
        self.voice.speech_synthesis(config.confirmCF)
        self.read_cf(cf)
        self.voice.speech_synthesis(config.confirm)
        if self.confirm():
            self.page.current_page.buttonConferma.invoke()
            self.user_page()
        else:
            self.page.current_page.buttonIndietro.invoke()
            self.recognition_page()

    def user_page(self):
        self.voice.speech_synthesis(self.page.current_page.name.cget("text"))
        self.voice.speech_synthesis(self.page.current_page.patient.cget("text"))
        self.voice.speech_synthesis("Codice fiscale: ")
        self.read_cf(self.page.current_page.cf.cget("text").split("CODICE FISCALE: ")[1])
        self.voice.speech_synthesis("Delegati")
        count = 1
        for delegateLabel in self.page.current_page.delegatesLabels:
            if delegateLabel.cget("text") is not "-":
                self.voice.speech_synthesis("Delegato numero "+count)
                self.read_cf(delegateLabel.cget("text"))
                count +=1
        if count == 1:
            self.voice.speech_synthesis("Nessuno")
        self.voice.speech_synthesis("Farmaci ")
        for medicine in self.page.current_page.entries:
            if medicine.cget("text").__contains__(" x "):
                name_medicine = medicine.cget("text").split(" x ")[0]
                nbox = medicine.cget("text").split(" x ")[1]
                self.voice.speech_synthesis(name_medicine)
                self.voice.speech_synthesis("Numero di scatole "+nbox)
            else:
                self.voice.speech_synthesis(medicine.cget("text"))
        while count < 100000000:
            count += 1
        self.page.current_page.homeButton.invoke()
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
    #voice.speech_synthesis(config.initialMessage + " " + config.choice1 + " " + config.choice2)
    #choice = voice.speech_recognize()
    #text = config.initialMessage + " " + config.choice1 + " " + config.choice2
    #app = Pages.Page()
    #app.geometry('300x550')
    #vocal_app = VocalPages(app)
    #task = threading.Thread(target=vocal_app.start_page)
    #task.start()
    #voice.speech_synthesis(config.messageMedicine)
    #medicine = voice.speech_recognize()
    #correct_word = voice.medicine_autocorrect(medicine)
    #print("La medicina corretta è", correct_word)

