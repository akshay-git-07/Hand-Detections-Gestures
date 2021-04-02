import cv2
import HandTrackingModule

# img = cv2.imread('resources/fire.png')
#
# scale_percent = 15  # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
#
# fire = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# cv2.imshow('Image', fire)
# cv2.waitKey(0)

fire = cv2.imread('resources/doctor-strange.png')

detector = HandTrackingModule.HandDetector()
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img = detector.findHand(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    if lmList:
        # print(lmList[5], lmList[17], lmList[0])

        # x1, y1 = lmList[5][1], lmList[5][2]
        # x2, y2 = lmList[17][1], lmList[17][2]
        # x3, y3 = lmList[0][1], lmList[0][2]

        # cx, cy = (x1 + x2 + x3) // 3, (y1 + y2 + y3) // 3
        cx, cy = lmList[9][1], lmList[9][2]
        # cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        y1, y2 = cy - 90, cy - 90 + fire.shape[0]
        x1, x2 = cx - 90, cx - 90 + fire.shape[1]

        alpha_s = fire[:, :, 2] / 255.0
        alpha_l = 1.0 - alpha_s

        try:
            for c in range(0, 3):
                img[y1:y2, x1:x2, c] = (alpha_s * fire[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
        except ValueError as e:
            print(e)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
