import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import MR
import pySaliencyMapRunner
from flow import getFlow
from saliency import get_saliency_ft, get_saliency_rbd
from saliency_mbd import get_saliency_mbd   

if __name__ == "__main__":
    filename = sys.argv[1]
    mr = MR.MR_saliency()
    cap = cv2.VideoCapture("../videos/" + filename)

    ret, frame1 = cap.read()
    prvs = cv2.resize(frame1, (420,240))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        next = cv2.resize(frame, (420,240))
        ang, mag = getFlow(prvs, next)
        sal = mr.saliency(next)
        pSMR = pySaliencyMapRunner.getSalMask(next)
        rbd = get_saliency_rbd(next).astype('uint8')
        mbd = get_saliency_mbd(next).astype('uint8')
        cv2.imshow("rbd", rbd)
        cv2.imshow("img", next)
        cv2.imshow("mbd", mbd)
        cv2.imshow("pSMR", pSMR)
        cv2.imshow("MR", sal)
        cv2.imshow("ang", ang)
        cv2.imshow("mag", mag)
        cv2.waitKey(1)
        prvs = next