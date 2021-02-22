import cv2

messageCF = "Inserire codice fiscale: "
messageError = "Codice fiscale non valido"
cf = ""

def collect_data():
    global cf
    cf = input(messageCF)
    error = False
    if len(cf) != 16:
        print(messageError)
        error = True
    return error

def recognize():
    print("Recognize")