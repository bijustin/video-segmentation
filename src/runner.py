import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import MR
import pySaliencyMapRunner
from flow import getFlow
from saliency import get_saliency_ft, get_saliency_rbd
from saliency_mbd import get_saliency_mbd   
from skimage.measure import label  

def threshold(mask):
    blur = cv2.GaussianBlur(mask,(5,5),0)
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def getConsensus(masks):
    numpix = []
    for mask in masks:
        #print(len(numpix))
        #assert(((mask==0) | (mask==1)).all())
        numpix.append(np.sum(mask))
    numpix = np.array(numpix)
    q1 = np.percentile(numpix, 25)
    q2 = np.percentile(numpix, 50)
    q3 = np.percentile(numpix, 75)
    o1 = q1-1.5*(q3-q1)
    o3 = q3+1.5*(q3-q1)
    outlier_scale = []
    for total in numpix:
        if total < q2:
            tmp = (total-o1)/(q2-o1)
            outlier_scale.append(max(tmp, 0))
        else:
            tmp = (total-o3)/(q2-o3)
            outlier_scale.append(max(tmp, 0))
    totalweight = sum(outlier_scale)
    out = np.zeros_like(masks[0]).astype(np.float)
    for weight, img in zip(outlier_scale, masks):
        out +=weight*img
    print(out)
    out /= totalweight
    print(out)
    final = out > 0.5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    final = cv2.morphologyEx(final.astype(np.float), cv2.MORPH_OPEN, kernel)
    return getLargestCC(final)
    


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
        masks = []
        ang, mag = getFlow(prvs, next)
        masks.append(ang)
        masks.append(mag)
        #sal = mr.saliency(next)
        pSMR = pySaliencyMapRunner.getSalMask(next)
        masks.append(pSMR/255)
        rbd = get_saliency_rbd(next).astype('uint8')
        masks.append(threshold(rbd))
        mbd = get_saliency_mbd(next).astype('uint8')
        masks.append(threshold(mbd))
        final = getConsensus(masks)
        print(final)
        drawimg = next.copy()
        mask = np.zeros_like(drawimg)
        mask[:,:,0] = final.astype(np.float)*255
        drawimg = cv2.add(drawimg, mask)
        cv2.imshow("mask", final.astype(np.float))
        cv2.imshow("rbd", threshold(rbd))
        cv2.imshow("img", drawimg)
        cv2.imshow("mbd", threshold(mbd))
        cv2.imshow("pSMR", pSMR)
        #cv2.imshow("MR", sal)
        cv2.imshow("ang", ang)
        cv2.imshow("mag", mag)
        cv2.waitKey(1)
        prvs = next