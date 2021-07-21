import speech_recognition as sr
import pyttsx3


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
            print("Sono in ascolto... parla pure!")
            audio = self.recognizer_instance.listen(source)
            print("Ok! sto ora elaborando il messaggio!")
        try:
            text = self.recognizer_instance.recognize_google(audio, language="it-IT")
            print("Google ha capito: \n", text)
            return text
        except Exception as e:
            print(e)
            return None

    def speech_synthesis(self, text):
        self.synthesis.say(text)
        self.synthesis.runAndWait()


if __name__ == '__main__':
    voice = Voice()
    voice.speech_recognize()
    voice.speech_synthesis("Prova?")
