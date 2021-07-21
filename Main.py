from Pages import Page
from VoiceService import VocalPages
import threading


if __name__ == '__main__':
    app = Page()
    app.geometry('300x550')
    vocal_app = VocalPages(app)
    threading.Thread(target=vocal_app.startPage).start()
    threading.Thread(target=app.mainloop()).start()