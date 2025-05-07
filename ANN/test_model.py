import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

lstResult = ['Den Vau', 'SonTung', 'Jenni', 'Lisa', 'Truong Giang']

models = load_model('model.h5')

frame = cv2.imread('test_model/2.jpg') 

faces = face_detector.detectMultiScale(frame, 1.3, 5)

for (x, y, w, h) in faces:
    roi = cv2.resize(frame[y:y+h, x:x+w], (128, 128))
    result = np.argmax(models.predict(roi.reshape((-1, 128, 128, 3))))
    cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 50), 2)
    cv2.putText(frame, lstResult[result], (x-15, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 25, 255), 2)


cv2.imshow('FRAME', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()