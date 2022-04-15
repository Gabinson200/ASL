# sourcery skip: use-fstring-for-concatenation

from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
import progressbar
import mediapipe
import cv2
import time
import copy
import keyboard


def num_to_letter(numbers):
    sentence = ""
    letters = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "del",
        "nothing",
        "space",
    ]
    for i in range(len(numbers)):
        if numbers[i] == "del":
            sentence = sentence.rstrip(sentence[-1])
        elif numbers[i] == "nothing":
            break
        elif numbers[i] == "space":
            sentence += " "
        else:
            sentence += letters[numbers[i]]

    return sentence


model = keras.models.load_model("CNN_model.h5")


cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
time_curr = time.time()
while not keyboard.is_pressed("q"):
    __s, img = cap.read()
    tmp = copy.copy(img)
    hand, imgP = detector.findHands(img)

    if hand and len(hand) == 1 and time.time() >= time_curr + 1:
        img2 = tmp[
            hand[0]["bbox"][1] - 50
            if hand[0]["bbox"][1] > 50
            else hand[0]["bbox"][1] : hand[0]["bbox"][1] + hand[0]["bbox"][3] + 50
            if hand[0]["bbox"][1] + hand[0]["bbox"][3] + 50 < img.shape[0]
            else hand[0]["bbox"][1] + hand[0]["bbox"][3],
            hand[0]["bbox"][0] - 50
            if hand[0]["bbox"][0] > 50
            else hand[0]["bbox"][1] : hand[0]["bbox"][0] + hand[0]["bbox"][2] + 50
            if hand[0]["bbox"][0] + hand[0]["bbox"][2] + 50 < img.shape[1]
            else hand[0]["bbox"][0] + hand[0]["bbox"][2],
        ]
        try:
            final_img = cv2.resize(img2, (64, 64))
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            final_img = np.expand_dims(final_img, axis=0)
            cv2.imshow("Result", img2)
            pred = model.predict(final_img)
            test_val = np.argmax(pred, axis=1)
            print(num_to_letter(test_val))
            time_curr = time.time()
        except:
            pass

    cv2.imshow("Image", imgP)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
