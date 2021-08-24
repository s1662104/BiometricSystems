import speech_recognition as sr
import pyttsx3
import tkinter as tk
import config
import Pages
from difflib import SequenceMatcher
import pandas as pd
import Levenshtein
import time

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
        # state = 0 -> interact with the user; = 1 -> listening silently the user
        self.state = 0

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
                if self.state == 0:
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
                if dist < min_dist:
                    min_dist = dist
                    word = m
            return word


class VocalPages:
    def __init__(self, page: Pages.Page):
        self.page = page
        self.voice = Voice()
        self.page.change_widget_state()
        # state = 0 significa che il sistema di interfaccia vocale e' attivo; = 1 viceversa

    # -------------- Pages --------------

    def start_page(self, repeat=False):
        self.page.current_page.update_widget_state(self.page)
        if not repeat:
            self.voice.speech_synthesis(config.initialMessage + " " + config.choice1 + " " + config.choice2)
            # time.sleep(18)
        choice = self.check_command()
        if self.voice.compare_strings(choice, config.choice1.lower()):
            self.voice.speech_synthesis("Operazione scelta " + config.choice1)
            self.invoke_button(self.page.current_page.button1)
            self.enroll_page_CF()
        elif self.voice.compare_strings(choice, config.choice2.lower()):
            self.voice.speech_synthesis("Operazione scelta " + config.choice2)
            self.invoke_button(self.page.current_page.button2)
            self.recognition_choice_page()
        else:
            self.voice.speech_synthesis("Scegli tra: " + config.choice1 + " " + config.choice2)
            self.start_page(True)

    def enroll_page_CF(self):
        cf = self.page_CF()
        self.set_text_entry(self.page.current_page.entryCF, cf)
        self.enroll_page_name()

    def page_CF(self, first_time=True):
        # se l'entry e' gia' modificata, significa che l'utente ha proceduto con l'operazione e poi e' tornato indietro
        # quindi il sistema gli chiedera' conferma sul codice fiscale
        print(self.page.current_page.entryCF.get(),
              self.page.current_page.entryCF.get() == config.messageCF)
        if self.page.current_page.entryCF.get() == config.messageCF:
            if first_time:
                self.voice.speech_synthesis(config.messageCF + "\n Ricorda di fare lo spelling e di "
                                                               "dire una parola alla volta")
            else:
                self.voice.speech_synthesis(config.messageCF)
            cf = ""
            while len(cf) < 16:
                text = self.check_command(True)
                cf += self.spelling(text)
        else:
            cf = self.page.current_page.entryCF.get()
        if len(cf)!=16:
            self.set_text_entry(self.page.current_page.entryCF, config.messageCF)
            return self.page_CF(False)
        self.voice.speech_synthesis(config.confirmCF)
        self.read_cf(cf)
        self.voice.speech_synthesis(config.confirm)
        if self.confirm():
            return cf
        else:
            self.set_text_entry(self.page.current_page.entryCF,config.messageCF)
            return self.page_CF(False)

    def enroll_page_name(self):
        # si divide tra nome e cognome per riconoscere correttamente i casi di doppi nomi e cognomi composti
        if self.page.current_page.entryName.get() == config.messageN:
            self.voice.speech_synthesis(config.messageN)
            first_name = self.check_command()
            self.voice.speech_synthesis(config.messageC)
            last_name = self.check_command()
            self.voice.speech_synthesis(first_name + " " + last_name + " " + config.confirm)
            if self.confirm():
                self.set_text_entry(self.page.current_page.entryName,first_name + " " + last_name)
                self.voice.speech_synthesis(config.messagePhoto)
                self.invoke_button(self.page.current_page.invio)
                self.data_enrollment_page()
            else:
                first_name = self.check_name(first_name)
                last_name = self.check_name(last_name)
                self.set_text_entry(self.page.current_page.entryName,first_name + " " + last_name)
                self.voice.speech_synthesis(config.messagePhoto)
                self.invoke_button(self.page.current_page.invio)
                self.data_enrollment_page()
        else:
            self.voice.speech_synthesis(self.page.current_page.entryName.get() + " " + config.confirm)
            if self.confirm():
                self.voice.speech_synthesis(config.messagePhoto)
                self.invoke_button(self.page.current_page.invio)
                self.data_enrollment_page()
            else:
                self.set_text_entry(self.page.current_page.entryName,config.messageN)
                self.enroll_page_name()

    def data_enrollment_page(self, first_time=True):
        # se la prima entry della lista dei delegati è vuota
        if first_time:
            # chiedo se si vogliono aggiungere dei delegati
            self.voice.speech_synthesis(config.messageDelegate)
            if self.confirm():
                # aggiungo i delegati
                self.addDelegates()
        # altrimenti, dato che era gia' stata compilata
        else:
            # chiedo conferma dei codici fiscali inseriti appartenenti ai delegati
            for i in range(3):
                # se l'entry è stata compliata, chiedo conferma
                if self.page.current_page.delegateEntry[i].get() != "":
                    cf = self.page.current_page.delegateEntry[i].get()
                    self.voice.speech_synthesis(config.confirmCFDelegate + cf + "?")
                    self.read_cf(cf)
                    # se è errato, allora viene modificato
                    if not self.confirm():
                        self.change_delegate(i)
                else:
                    # se l'entry è vuota, chiedo se si vogliono aggiungere altri delegati
                    self.voice.speech_synthesis(config.continueToAddDelegate)
                    if self.confirm():
                        self.addDelegates(nDelegates=i)
                    break

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
        self.voice.speech_synthesis(config.numberMedicinesConfirm + str(num_medicines) + "?")
        if not self.confirm():
            self.data_enrollment_page()     # se la risposta è negativa, allora chiedo nuvamente il numero
        # altrimenti vado avanti ed aggiungo un numero di entry pari al numero appena inserito
        self.set_text_entry(self.page.current_page.entryNMedicine,num_medicines)
        self.invoke_button(self.page.current_page.buttonInvia)

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
            # Chiedo conferma della medicina appena dichiarata
            self.voice.speech_synthesis(config.medicineConfirm + entryMedicine + "?")
            # se il farmaco è errato, viene chiesto nuovamente di inserirlo
            if not self.confirm():
                # se era gi stato inserito un farmco, ma era errato, viene cancellato
                if not self.page.current_page.medicineEntry[i].get() == "":
                    self.page.current_page.medicineEntry[i].delete(0, tk.END)
                    self.page.current_page.medicineEntry[i].insert(0, "")
                continue
            dosaggioCorretto = False
            while not dosaggioCorretto:
                # chiedo il dosaggio per il farmaco
                self.voice.speech_synthesis(config.dosageMedicine)
                dosaggio = self.check_command()
                # chiedo conferma del dosaggio
                self.voice.speech_synthesis("Confermi che" + dosaggio + "sia il dosaggio corretto?")
                if self.confirm():
                    self.page.current_page.medicineEntry[i].insert(0, entryMedicine + " " + dosaggio)
                    i += 1
                    dosaggioCorretto = True

        self.voice.speech_synthesis(config.medConfirm)
        # Se c'è quale medicinale errato, chiedo qual è e lo correggo
        if not self.confirm():
            self.change_medicine()

        self.invoke_button(self.page.current_page.buttonConferma)
        self.information_page()

    def information_page(self):
        self.voice.speech_synthesis(self.page.current_page.label.cget("text"))
        time.sleep(5)
        self.invoke_button(self.page.current_page.homeButton)
        self.start_page()

    def recognition_choice_page(self):
        self.voice.speech_synthesis("Scegli tra " + " " + config.recognitionChoice1 + " " +
                                    config.recognitionChoice2)
        text = self.check_command()
        if self.voice.compare_strings(text, config.recognitionChoice1.lower()):
            self.voice.speech_synthesis("Ruolo scelto: " + config.recognitionChoice1)
            self.invoke_button(self.page.current_page.button1)
            self.recognition_page()
        elif self.voice.compare_strings(text, config.recognitionChoice2.lower()):
            self.voice.speech_synthesis("Ruolo scelto: " + config.recognitionChoice2)
            self.invoke_button(self.page.current_page.button2)
            self.recognition_page()
        else:
            self.recognition_choice_page()

    def recognition_page(self, indietro=False):
        cf = self.page_CF(indietro)
        self.set_text_entry(self.page.current_page.entryCF, cf)
        self.voice.speech_synthesis(config.messagePhoto)
        self.invoke_button(self.page.current_page.buttonInvia)
        self.data_recognition_page()

    def data_recognition_page(self):
        cf = self.page.current_page.cf.cget("text").split("CODICE FISCALE: ")[1]
        self.voice.speech_synthesis(config.confirmCF)
        self.read_cf(cf)
        self.voice.speech_synthesis(config.confirm)
        if self.confirm():
            self.invoke_button(self.page.current_page.buttonConferma)
            self.user_page()
        else:
            self.invoke_button(self.page.current_page.backButton)
            self.recognition_page()

    def user_page(self):
        print(self.page.current_page)
        self.voice.speech_synthesis(self.page.current_page.name.cget("text"))
        self.voice.speech_synthesis(self.page.current_page.patient.cget("text"))
        self.voice.speech_synthesis("Codice fiscale: ")
        self.read_cf(self.page.current_page.cf.cget("text").split("CODICE FISCALE: ")[1])
        self.voice.speech_synthesis("Delegati")
        count = 1
        for delegateLabel in self.page.current_page.delegatesLabels:
            if delegateLabel.cget("text") is not "-":
                self.voice.speech_synthesis("Delegato numero " + count)
                self.read_cf(delegateLabel.cget("text"))
                count += 1
        if count == 1:
            self.voice.speech_synthesis("Nessuno")
        self.voice.speech_synthesis("Farmaci ")
        for medicine in self.page.current_page.entries:
            if medicine.cget("text").__contains__(" x "):
                name_medicine = medicine.cget("text").split(" x ")[0]
                nbox = medicine.cget("text").split(" x ")[1]
                self.voice.speech_synthesis(name_medicine)
                self.voice.speech_synthesis("Numero di scatole " + nbox)
            else:
                self.voice.speech_synthesis(medicine.cget("text"))
        time.sleep(10)
        self.invoke_button(self.page.current_page.homeButton)
        self.start_page()

    # -------------- Functions --------------
    def addDelegates(self, nDelegates=0):
        while True:
            # chiedo di fare lo spelling del codice fiscale
            self.voice.speech_synthesis(config.numberDelegate)
            cf = ""
            while len(cf) < 16:
                text = self.check_command(True)
                cf += self.spelling(text)

            print("Lunghezza codice fiscale:",len(cf))

            # se la stringa del cf non ha la lunghezza corretta, si ripete da capo lo spelling
            if len(cf) != 16:
                self.voice.speech_synthesis(config.cfDeleateError)
                continue

            # chiedo conferma del codice fiscale del delegato
            self.voice.speech_synthesis(config.confirmCFDelegate+cf+"?")
            self.read_cf(cf)

            # se il codice fiscale è giusto
            if self.confirm():
                # inserisco il cf nell'entry
                self.set_text_entry(self.page.current_page.delegateEntry[nDelegates], cf)
                nDelegates += 1
                # se si è raggiunto il numero massimo di delegati,allora termino
                if nDelegates == 2:
                    self.voice.speech_synthesis(config.numberMaxDelegate)
                    break
                # se possono esserne inseriti altri, chiedo se ce ne sono da aggiungere
                self.voice.speech_synthesis(config.continueToAddDelegate)
                # se non ci sono altri delegati da inserire, allora termino
                if not self.confirm():
                    break
        return

    def change_delegate(self, index):



        return

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
                    self.set_text_entry(self.page.current_page.medicineEntry[pos],entryMedicine)
                    print(entryMedicine)
                else:
                    continue
                # Chiedo se ci sono altri farmaci da cambiare
                self.voice.speech_synthesis(config.otherMedToChange)
                if not self.confirm():
                    return
            else:
                self.voice.speech_synthesis(config.errorSpeechRec)

    def check_name(self, text, repeat=False):
        if not repeat:
            self.voice.speech_synthesis(config.errorFirstName + " " + text)
            if self.confirm():
                return text
            else:
                return self.check_name(text, True)
        else:
            self.voice.speech_synthesis(config.errorSpelling)
            nameSpelled = self.check_command(True)
            name = self.spelling(nameSpelled, True)
            print(name)
            self.voice.speech_synthesis(name + " " + config.confirm)
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
        return self.voice.compare_strings(text, config.yes.lower())

    def read_cf(self, cf):
        self.voice.synthesis.setProperty('rate', 100)
        spelling = ""
        for c in cf:
            if c.isalpha():
                spelling += spell[c] + " "
            else:
                spelling += c + " "
        self.voice.speech_synthesis(spelling)
        self.voice.synthesis.setProperty('rate', 140)

    def check_command(self, higher_pause=False):
        text = self.voice.speech_recognize(higher_pause)
        if self.voice.state == 0 and self.voice.compare_strings(text, "Indietro"):
            no_back_pages = [self.page.get_pages()[Pages.StartPage], self.page.get_pages()[Pages.InformationPage],
                             self.page.get_pages()[Pages.UserPage]]
            if self.page.current_page in no_back_pages:
                self.voice.speech_synthesis(config.messageError)
                return self.check_command(higher_pause)
            else:
                self.invoke_button(self.page.current_page.backButton)
                self.go_to_current_page_function()
        elif self.voice.state == 0 and self.voice.compare_strings(text, "stop"):
            self.voice.speech_synthesis(config.messageAfterStop)
            self.page.change_widget_state()
            self.page.current_page.update_widget_state(self.page)
            self.voice.state = 1
            while True:
                self.check_command()
        elif self.voice.state == 1 and self.voice.compare_strings(text, "start"):
            self.page.change_widget_state()
            self.page.current_page.update_widget_state(self.page)
            self.voice.state = 0
            self.go_to_current_page_function()
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
            self.data_enrollment_page(first_time=False)
        elif self.page.current_page.__class__ is Pages.DataRecognitionPage:
            self.data_recognition_page()
        elif self.page.current_page.__class__ is Pages.UserPage:
            self.user_page()

    def set_text_entry(self,entry, text):
        entry["state"] = "normal"
        entry.delete(0, tk.END)
        entry.insert(0, text)
        entry["state"] = "disabled"

    def invoke_button(self, button):
        button["state"] = "normal"
        button.invoke()
        button["state"] = "disabled"

if __name__ == '__main__':
    voice = Voice()
    # for c in "SMRLPG34R92H784R":
    #     if c.isalpha():
    #         voice.speech_synthesis(spell[c])
    #     else:
    #         voice.speech_synthesis(c)
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
