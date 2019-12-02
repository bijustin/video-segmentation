import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from pyflow import pyflow

def getFlow(prev, next):
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    u, v, im2W = pyflow.coarse2fine_flow(
            prev.astype(float)/255, next.astype(float)/255, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    q1ang = np.percentile(ang, 25)
    q3ang = np.percentile(ang, 75)
    o1ang = q1ang-1.5*(q3ang-q1ang)
    o3ang = q3ang+1.5*(q3ang-q1ang)
    idxang = np.logical_or((ang > o3ang), (ang< o1ang))
    ang[idxang] = 1
    ang[np.invert(idxang)] = 0

    q1mag = np.percentile(mag, 25)
    q3mag = np.percentile(mag, 75)
    o1mag = q1mag-1.5*(q3mag-q1mag)
    o3mag = q3mag+1.5*(q3mag-q1mag)
    idxmag = np.logical_or((mag > o3mag), (mag < o1mag))
    mag[idxmag] = 1
    mag[np.invert(idxmag)] = 0


    return ang, mag

