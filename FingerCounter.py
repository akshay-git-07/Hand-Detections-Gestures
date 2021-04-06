import cv2
import time
import os
import HandTrackingModule

width, height = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

folderPath = 'resources/fingers'
myList = os.listdir(folderPath)
# print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

prevTime = 0
detector = HandTrackingModule.HandDetector(detectionCon=0.7)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHand(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        fingers = []
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)

        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]

        # cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        # cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 25)

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(img, f'FPS: {(int(fps))}', (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
