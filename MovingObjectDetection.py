import cv2

firstFrame = None
Area = 700

cam = cv2.VideoCapture(0)

while True:
    _,img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImg = cv2.resize(grayImg, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
    gaussianImg = cv2.GaussianBlur(grayImg, (21,21), 0)
    cv2.imshow(gaussianImg)
    if firstFrame is None:
        firstFrame = gaussianImg
        continue
    isImg = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(isImg, 10, 200, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)
    #stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    #disparity = stereo.compute(img, grayImg)
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    