import cv2
import numpy as np
import sys
import os
from numpy.linalg import svd
from matplotlib import pyplot as plt

def convertToConcantenate(vidFrame):
    a = cv2.cvtColor(vidFrame, cv2.COLOR_BGR2GRAY)
    b = cv2.resize(a, (420, 240))
    c = np.ndarray.flatten(b)
    return c

def rankedRowToMask(vidFrame):
    vidFrame = np.resize(vidFrame, (240, 420))
    vidFrame = cv2.normalize(vidFrame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(
        np.uint8)
    #_, vidFrame = cv2.threshold(vidFrame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return vidFrame

def getVid(vidFile):
    filename = vidFile
    cap = cv2.VideoCapture(filename)
    imageCounter = 0
    frameCounter = 1
    ret, firstFrame = cap.read()
    compositeFrames = convertToConcantenate(firstFrame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rowFrame = convertToConcantenate(frame)
        compositeFrames = np.row_stack((compositeFrames, rowFrame))
        if frameCounter == 4:
            u, s, vh = np.linalg.svd(compositeFrames, full_matrices=False)
            rankApprox = np.dot(u * s, vh)
            Ub = u[:, 0:0]
            Sb = s[0]
            Vb = vh[0:0, :]
            back = np.dot(Ub * Sb, Vb)
            Uf = u[:, 1:7]
            Sf = s[1:7]
            Vf = vh[1:7, :]
            fore = np.dot(Uf * Sf, Vf)
            frameCounter = 0
            compositeFrames = convertToConcantenate(frame)
            for x in range(0, 4):
                testFrame = rankedRowToMask(np.abs(fore[x:] - back[x:]))
                testFrame = cv2.GaussianBlur(testFrame, (5, 5), 0)
                #testFrame = cv2.blur(testFrame, (5, 5))
                #cv2.imwrite("img%d.jpg" % imageCounter, testFrame)
                #if imageCounter == 0:
                #    vid = testFrame
                #imageCounter = imageCounter + 1
                #vid = np.dstack((vid, testFrame))
                cv2.imshow("vid", testFrame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        frameCounter = frameCounter + 1
    return 0

def getFrame(pastFrame,presentFrame):
    compositeFrames = convertToConcantenate(pastFrame)
    rowFrame = convertToConcantenate(presentFrame)
    compositeFrames = np.row_stack((compositeFrames, rowFrame))
    u, s, vh = np.linalg.svd(compositeFrames, full_matrices=False)
    rankApprox = np.dot(u * s, vh)
    Ub = u[:, 0:0]
    Sb = s[0]
    Vb = vh[0:0, :]
    back = np.dot(Ub * Sb, Vb)
    Uf = u[:, 1:7]
    Sf = s[1:7]
    Vf = vh[1:7, :]
    fore = np.dot(Uf * Sf, Vf)
    testFrame = rankedRowToMask(np.abs(fore[1:] - back[1:]))
    return testFrame

if __name__ == "__main__":
     #vidName = sys.argv[1]
     #video = getVid(vidName)
     video = getVid(0)
     cv2.destroyAllWindows()