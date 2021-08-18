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

positions = {"il primo": 0, "il secondo": 1, "il terzo": 2, "il quarto": 3, "il quinto": 4, "il sesto": 5,
                "il settimo": 6, "l'ottavo": 7, "il nono": 8, "il decimo": 9}

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

    # -------------- Pages --------------

    def start_page(self, repeat=False):
        if not repeat:
            self.voice.speech_synthesis(config.initialMessage + " " + config.choice1 + " " + config.choice2)
            # time.sleep(18)
        choice = self.check_command()
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
        cf = self.page_CF()
        self.page.current_page.entryCF.delete(0, tk.END)
        self.page.current_page.entryCF.insert(0, cf)
        self.enroll_page_name()

    def page_CF(self, first_time=True):
        # se l'entry e' gia' modificata, significa che l'utente ha proceduto con l'operazione e poi e' tornato indietro
        # quindi il sistema gli chiedera' conferma sul codice fiscale
        if self.page.current_page.entryCF.get() == config.messageCF:
            if first_time:
                self.voice.speech_synthesis(config.messageCF+"\n Ricorda di fare lo spelling e di "
                                                             "dire una parola alla volta")
            else:
                self.voice.speech_synthesis(config.messageCF)
            cf = ""
            # TODO CONTROLLO SULLA LUNGHEZZA DEL CF
            while len(cf) < 16:
                text = self.check_command(True)
                cf += self.spelling(text)
        else:
            cf = self.page.current_page.entryCF.get()
        self.voice.speech_synthesis(config.confirmCF)
        self.read_cf(cf)
        self.voice.speech_synthesis(config.confirm)
        if self.confirm():
            return cf
        else:
            self.page.current_page.entryCF.delete(0, tk.END)
            self.page.current_page.entryCF.insert(0, config.messageCF)
            return self.page_CF(False)

    def enroll_page_name(self):
        # si divide tra nome e cognome per riconoscere correttamente i casi di doppi nomi e cognomi composti
        if self.page.current_page.entryName.get() == config.messageN:
            self.voice.speech_synthesis(config.messageN)
            first_name = self.check_command()
            self.voice.speech_synthesis(config.messageC)
            last_name = self.check_command()
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
        else:
            self.voice.speech_synthesis(self.page.current_page.entryName.get() + " " + config.confirm)
            if self.confirm():
                self.voice.speech_synthesis(config.messagePhoto)
                self.page.current_page.invio.invoke()
                self.data_enrollment_page()
            else:
                self.page.current_page.entryName.delete(0, tk.END)
                self.page.current_page.entryName.insert(0, config.messageN)
                self.enroll_page_name()

    def data_enrollment_page(self):

        print(self.page.current_page.entryNMedicine.get())

        # se l'entry non e' modificata, significa che l'utente non ha inserito il numero dei farmaci
        if self.page.current_page.entryNMedicine.get() == "":
            self.voice.speech_synthesis(config.numberMedicines)
            num_string = self.check_command()

            if numbers.__contains__(num_string):
                num_medicines = numbers.__getitem__(num_string)
            else:
                try:
                    num_medicines = int(num_string)
                except Exception as e:
                    self.voice.speech_synthesis(config.messageError)
                    return self.data_enrollment_page()
        # se l'entry e' modificata, l'utente aveva gia' inserito il numero dei farmaci e poi era tornato indietro
        else:
            num_medicines = self.page.current_page.entryNMedicine.get()

        # chiedo conferma del numero dei farmaci
        self.voice.speech_synthesis(config.numberMedicinesConfirm + num_medicines + "?")
        if not self.confirm():
            self.data_enrollment_page()     # se la risposta è negativa, allora chiedo nuvamente il numero
        # altrimenti vado avanti ed aggiungo un numero di entry pari al numero appena inserito
        self.page.current_page.entryNMedicine.delete(0, tk.END)
        self.page.current_page.entryNMedicine.insert(0, num_medicines)
        self.page.current_page.buttonInvia.invoke()

        i = 0

        # ogni entry deve essere controllato per vedere se è stato modificato o meno
        while i < num_medicines:

            # se non è presente nessun nome di un farmaco, allora viene richiesto all'untente di aggiungerlo
            if self.page.current_page.medicineEntry[i].get() == "":
                self.voice.speech_synthesis(config.messageMedicine)
                entryMedicine = self.check_command()
                entryMedicine = self.voice.medicine_autocorrect(entryMedicine)
            else:
                entryMedicine = self.page.current_page.medicineEntry[i].get()
            #Chiedo conferma della medicina appena dichiarata
            self.voice.speech_synthesis(config.medicineConfirm+entryMedicine+"?")
            if self.confirm():
                self.page.current_page.medicineEntry[i].insert(0, entryMedicine)
                print(entryMedicine)
                i += 1

        self.voice.speech_synthesis(config.medConfirm)
        #Se c'è quale medicinale errato, chiedo qual è e lo correggo
        if not self.confirm():
            self.change_medicine()

        self.page.current_page.buttonConferma.invoke()
        self.information_page()


    def information_page(self):
        self.voice.speech_synthesis(self.page.current_page.label.cget("tex"))
        count = 0
        while count < 10000000:
            count += 1
        self.page.current_page.homeButton.invoke()
        self.start_page()

    def recognition_choice_page(self):
        self.voice.speech_synthesis("Scegli tra "+" "+config.recognitionChoice1+" "+
                                    config.recognitionChoice2)
        text = self.check_command()
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

    def recognition_page(self, indietro = False):
        cf = self.page_CF(indietro)
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
            self.page.current_page.backButton.invoke()
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

    # -------------- Functions --------------

    def change_medicine(self):
        while True:
            # Chiedo quel è il farmaco che deve essere corretto
            self.voice.speech_synthesis(config.changeMed)
            index = self.check_command()
            # Se la posizione è stata capita correttamente, allora ottengo l'indice dell'array dei farmaci
            if positions.__contains__(index):
                pos = positions.__getitem__(index)
                if pos < 0 or pos > len(self.page.current_page.medicineEntry):
                    self.voice.speech_synthesis(config.errorIndexMedicine)
                    continue
                # Chiedo il nome del farmaco
                self.voice.speech_synthesis(config.messageMedicine)
                entryMedicine = self.check_command()
                entryMedicine = self.voice.medicine_autocorrect(entryMedicine)
                # Chiedo conferma della medicina appena dichiarata
                self.voice.speech_synthesis(config.medicineConfirm + entryMedicine + "?")
                # Se il farmaco è corretto, lo sostituisco con quello precedentemente inserito nella posizione indicata
                if self.confirm():
                    self.page.current_page.medicineEntry[pos].delete(0, tk.END)
                    self.page.current_page.medicineEntry[pos].insert(0, entryMedicine)
                    print(entryMedicine)
                else:
                    continue
                # Chiedo se ci sono altri farmaci da cambiare
                self.voice.speech_synthesis(config.otherMedToChange)
                if not self.confirm():
                    return
            else:
                self.voice.speech_synthesis(config.errorSpeechRec)

    def check_name(self,text,  repeat= False):
        if not repeat:
            self.voice.speech_synthesis(config.errorFirstName + " " + text)
            if self.confirm():
                return text
            else:
                return self.check_name(text,True)
        else:
            self.voice.speech_synthesis(config.errorSpelling)
            nameSpelled = self.check_command(True)
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
        text = self.check_command()
        print(text)
        return self.voice.compare_strings(text, config.yes.lower())

    def read_cf(self, cf):
        for c in cf:
            if c.isalpha():
                self.voice.speech_synthesis(spell[c])
            else:
                self.voice.speech_synthesis(c)

    def check_command(self, higher_pause=False):
        text = self.voice.speech_recognize(higher_pause)
        if self.voice.compare_strings(text,"Indietro"):
            no_back_pages = [self.page.get_pages()[Pages.StartPage], self.page.get_pages()[Pages.InformationPage],
                             self.page.get_pages()[Pages.UserPage]]
            if self.page.current_page in no_back_pages:
                self.voice.speech_synthesis(config.messageError)
                return self.check_command(higher_pause)
            else:
                self.page.current_page.backButton.invoke()
                self.go_to_current_page_function()
        elif self.voice.compare_strings(text,"Stop"):
            pass
        elif self.voice.compare_strings(text,"Start"):
            pass
        else:
            return text

    def go_to_current_page_function(self):
        if self.page.current_page.__class__ is Pages.StartPage:
            self.start_page()
        elif self.page.current_page.__class__ is Pages.EnrollmentPage:
            self.enroll_page_CF()
        elif self.page.current_page.__class__ is Pages.RecognitionChoicePage:
            self.recognition_choice_page()
        elif self.page.current_page.__class__ is Pages.RecognitionPage:
            self.recognition_page()
        elif self.page.current_page.__class__ is Pages.DataEnrollmentPage:
            self.data_enrollment_page()
        elif self.page.current_page.__class__ is Pages.DataRecognitionPage:
            self.data_recognition_page()
        elif self.page.current_page.__class__ is Pages.UserPage:
            self.user_page()


if __name__ == '__main__':
    voice = Voice()
    while True:
        # Chiedo quel è il farmaco che deve essere corretto
        voice.speech_synthesis(config.changeMed)
        index = voice.speech_recognize()
        # Se la posizione è stata capita correttamente, allora ottengo l'indice dell'array dei farmaci
        if positions.__contains__(index):
            pos = positions.__getitem__(index)
            # Chiedo il nome del farmaco
            voice.speech_synthesis(config.messageMedicine)
            entryMedicine = voice.speech_recognize()
            entryMedicine = voice.medicine_autocorrect(entryMedicine)
            # Chiedo conferma della medicina appena dichiarata
            voice.speech_synthesis(config.medicineConfirm + entryMedicine + "?")
            # Se il farmaco è corretto, lo sostituisco con quello precedentemente inserito nella posizione indicata
            text = voice.speech_recognize()
            if voice.compare_strings(text, config.yes.lower()):
                print(entryMedicine)
            else:
                continue
            # Chiedo se ci sono altri farmaci da cambiare
            voice.speech_synthesis(config.otherMedToChange)
            text = voice.speech_recognize()
            if not voice.compare_strings(text, config.yes.lower()):
                break
        else:
            voice.speech_synthesis(config.errorSpeechRec)
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

