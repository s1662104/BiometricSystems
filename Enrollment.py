messageCF = "Inserire codice fiscale: "
messageError = "Codice fiscale non valido"
messageN = "Inserire nome: "
cf = ""
nome = ""

def collect_info():
    global cf,nome
    cf = input(messageCF)
    error = False
    if len(cf) != 16:
        print(messageError)
        error = True
    if not error:
        nome = input(messageN)
    return error

def enroll():
    print("Enroll")