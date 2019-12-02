""" 
https://github.com/pathak22/pyflow
http://people.csail.mit.edu/celiu/OpticalFlow/
The repo is the wrapper for Ce Liu's Coarse2FineTwoFram method
"""

import cv2
import numpy as np
import pyflow


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

    img2 = im2double(frame2)
    
    vx, vy, _ = pyflow.coarse2fine_flow(img1, img2, alpha, ratio,minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations)


    # flow[:,:,1] = vx
    # flow[:,:,2] = vy
    flow = np.concatenate((vx[..., None], vy[..., None]), axis=2)

    return flow


def DenseOpticalFlow(filename):

    cap = cv2.VideoCapture("../../videos/" + filename)
    
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        flow = opticalFlow_siftFlow(frame1, frame2)

        hsv = np.zeros(frame1.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        

        cv2.imshow('frame2',rgb)
        cv2.waitKey(1)

        frame1 = frame2


if __name__ == "__main__":
    DenseOpticalFlow("car-turn.mp4")       
