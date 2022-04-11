# sourcery skip: use-fstring-for-concatenation
from cvzone.HandTrackingModule import HandDetector
import cv2
import time
import copy

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
time_curr = time.time()
while True:
    __s, img = cap.read()
    tmp = copy.copy(img)
    hand, imgP = detector.findHands(img)
    cv2.imshow("R", tmp)

    if hand and len(hand) == 1 and time.time() >= time_curr + 1:
        img2 = tmp[
            hand[0]["bbox"][1] - 20
            if hand[0]["bbox"][1] > 20
            else hand[0]["bbox"][1] : hand[0]["bbox"][1] + hand[0]["bbox"][3] + 20
            if hand[0]["bbox"][1] + hand[0]["bbox"][3] + 20 < img.shape[0]
            else hand[0]["bbox"][1] + hand[0]["bbox"][3],
            hand[0]["bbox"][0] - 20
            if hand[0]["bbox"][0] > 20
            else hand[0]["bbox"][1] : hand[0]["bbox"][0] + hand[0]["bbox"][2] + 20
            if hand[0]["bbox"][0] + hand[0]["bbox"][2] + 20 < img.shape[1]
            else hand[0]["bbox"][0] + hand[0]["bbox"][2],
        ]
        cv2.imshow("Result", img2)
        time_curr = time.time()
    cv2.imshow("Image", imgP)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
