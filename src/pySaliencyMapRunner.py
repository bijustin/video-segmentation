import cv2
import matplotlib.pyplot as plt
import pySaliencyMap

def getSalMask(img):
    imgsize = img.shape
    img_width  = imgsize[1]
    img_height = imgsize[0]
    
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
    return sm.SMGetBinarizedSM(img)