import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currentTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLandmark in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLandmark, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
