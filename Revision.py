import cv2
import imutils

FirstFrame = None
Area = 700

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    Text = "Normal"
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImg = cv2.resize(grayImg, None, fx= 1, fy= 1)
    gaussianImg = cv2.GaussianBlur(grayImg, (21,21), 0)
    cv2.imshow("Image", gaussianImg)
    if FirstFrame is None:
        FirstFrame = gaussianImg
        continue
    cv2.imshow('FirstFrame', FirstFrame)
    imgDiff = cv2.absdiff(FirstFrame, gaussianImg)
    cv2.imshow('Diff', imgDiff)
    threshImg = cv2.threshold(imgDiff, 10, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('thresh', threshImg)
    threshImg = cv2.dilate(threshImg, None, iterations= 2)
    cv2.imshow('thresh-dillate', threshImg)
    
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < Area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(img, cnts, -1, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        Text = 'Moving Object Detected'
    cv2.putText(img, Text, fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1, color=(0,255,0))
    cv2.imshow('Detection', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows