# sourcery skip: use-fstring-for-concatenation

from cvzone.HandTrackingModule import HandDetector
import numpy as np
import cv2
from tensorflow import keras
import cv2
import time
import copy
from autocorrect import Speller


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
        " delete ",
        " nothing ",
        " ",
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


model = keras.models.load_model("CNN_model_GOOD.h5")

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
time_curr = time.time()

output = "Predicted string: "

while True:
    __s, img = cap.read()
    tmp = copy.copy(img)
    hand, imgP = detector.findHands(img)

    if hand and len(hand) == 1 and time.time() >= time_curr + 1:
        if hand[0]["bbox"][3] > hand[0]["bbox"][2]:
            img2 = tmp[
                hand[0]["bbox"][1] - 50 : hand[0]["bbox"][1] + hand[0]["bbox"][3] + 100,
                hand[0]["bbox"][0] - 50 : hand[0]["bbox"][0] + hand[0]["bbox"][3] + 100,
            ]
        else:
            img2 = tmp[
                hand[0]["bbox"][1] - 50 : hand[0]["bbox"][1] + hand[0]["bbox"][2] + 100,
                hand[0]["bbox"][0] - 50 : hand[0]["bbox"][0] + hand[0]["bbox"][2] + 100,
            ]

        try:
            final_img = cv2.resize(img2, (64, 64), interpolation=cv2.INTER_AREA)
            final_img = np.array(final_img)
            final_img = final_img.reshape((-1, 64, 64, 3))
            cv2.imshow("Result", img2)
            pred = model.predict(final_img)
            test_val = np.argmax(pred, axis=1)
            time_curr = time.time()
            if output[-1:] != num_to_letter(test_val):
                output = output + num_to_letter(test_val)
            print(output)
        except Exception:
            print("Please bring hand closer to center")
    cv2.imshow("Image", imgP)
    if cv2.waitKey(1) == ord("."):
        output = "Predicted string: "
    if cv2.waitKey(1) == ord("q"):
        break


output = output[17:]
fin_str = ""
spell = Speller(lang="en")
for word in output.split():
    fin_str = fin_str + spell(word) + " "
print(f"Final prediction: {fin_str}")

cap.release()
cv2.destroyAllWindows()
