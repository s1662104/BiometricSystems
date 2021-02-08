import Database
import cv2
import numpy as np
from PIL import Image as im

if __name__ == '__main__':
    db = Database.Database()
    data = db.get_normalized_user(1)
    while(True):
        cv2.imshow('frame', data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

