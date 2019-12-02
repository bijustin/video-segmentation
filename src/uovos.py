""" 
Unsupervised Online Video Object Segmentation
with Motion Property Understanding

paper link: https://arxiv.org/pdf/1810.03783.pdf
github repo: https://github.com/visiontao/uovos
"""

import cv2
import numpy as np
import bob.ip.optflow.liu


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


def opticalFlow_siftFlow(frame1, frame2):

    flow = frame1

    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30

    img1 = im2double(frame1)
    # img1 = img1.tolist()

    img2 = im2double(frame2)
    # img2 = img2.tolist()
    
    vx, vy, _ = bob.ip.optflow.liu.sor.flow(img1, img2, alpha, ratio,minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations)


    flow[:,:,1] = vx
    flow[:,:,2] = vy

    return flow


def UOVOS(filename):

    cap = cv2.VideoCapture("../videos/" + filename)
    
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        flow = opticalFlow_siftFlow(frame2, frame1)
        cv2.imshow('frame2',flow)
        cv2.waitKey(1)

        frame1 = frame2


if __name__ == "__main__":
    UOVOS("bus.mp4")       
