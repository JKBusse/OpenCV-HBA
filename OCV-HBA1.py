import cv2
import time
import numpy as np
from datetime import datetime

path = "/Users/JohnDoe/Desktop/OpenCV/"

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
org1 = (50, 100)
orgRect = (0, 300)
orgRect1 = (0, 350)
fontScale = 1
color = (255, 0, 0)
thickness = 2
start_point = (5, 5)
end_point = (220, 220)
color = (255, 0, 0)
thicknessRect = -1
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



threshRed = 5000
threshWhite = 10000
#frame = cv2.imread("/Users/JohnDoe/Desktop/OpenCV/frame.png")
  
while(1):
    _, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180) 

    #frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 15)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lower_red = np.array([100,0,0])
    upper_red = np.array([168,86,86])

    lower_white = np.array([200,255,255])
    upper_white = np.array([255,255,255])

    maskRed = cv2.inRange(hsv, lower_red, upper_red)
    resultRed = cv2.bitwise_and(frame, frame, mask = maskRed)
    maskRedPixel = cv2.countNonZero(maskRed)

    maskWhite = cv2.inRange(hsv, lower_white, upper_white)
    resultWhite = cv2.bitwise_and(frame, frame, mask = maskWhite)
    maskWhitePixel = cv2.countNonZero(maskWhite)

    maskWhitePixelStr = "White: " + str(maskWhitePixel)
    maskRedPixelStr = "Red: " + str(maskRedPixel)

    maskRed = cv2.putText(maskRed, str(maskRedPixel), org, font, fontScale, color, thickness, cv2.LINE_AA)
    maskWhite = cv2.putText(maskWhite, str(maskWhitePixelStr), org, font, fontScale, color, thickness, cv2.LINE_AA)
    maskWhite = cv2.putText(maskWhite, str(maskRedPixelStr), org1, font, fontScale, color, thickness, cv2.LINE_AA)

    dt = datetime.now()
    ts = datetime.timestamp(dt)

    cv2.imshow('frame', frame)
    frame = cv2.resize(frame, (0,0), fx=0.2, fy=0.2) 
    pathFrame = path + "frame/frame" + str(dt) + ".png"
    cv2.imwrite(pathFrame, frame)

    cv2.imshow('maskRed', maskRed)
    cv2.imwrite("maskRed.png", maskRed)

    cv2.imshow('maskWhite', maskWhite)
    cv2.imwrite("maskWhite.png", maskWhite)
    if maskRedPixel < threshRed and maskWhitePixel < threshWhite:
        resultRed = cv2.rectangle(resultRed, start_point, end_point, (0,255,0), thicknessRect)
    else:
        resultRed = cv2.rectangle(resultRed, start_point, end_point, (0,0,255), thicknessRect)
    resultRed = cv2.putText(resultRed, str(maskWhitePixelStr), orgRect, font, fontScale, color, thickness, cv2.LINE_AA)
    resultRed = cv2.putText(resultRed, str(maskRedPixelStr), orgRect1, font, fontScale, color, thickness, cv2.LINE_AA)


    cv2.imshow('resultRed', resultRed)
    resultRed = cv2.resize(resultRed, (0,0), fx=0.5, fy=0.5) 
    pathResultRed = path + "resultRed/resultRed" + str(dt) + ".png"
    cv2.imwrite(pathResultRed, resultRed)
    
    cv2.waitKey(1)
    if cv2.waitKey(1) == 13:
        break
    #break
  
cv2.destroyAllWindows()
cap.release()
