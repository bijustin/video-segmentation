import cv2
import numpy as np
import sys
            

if __name__ == "__main__":
    filename = sys.argv[1]
    cap = cv2.VideoCapture(filename)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[:,:,1] = 255

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        next = cv2.GaussianBlur(next,(21,21),0)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag1 = mag.reshape(-1,1)
        mag1 = mag1[mag1 > 0.5]

       # mag1 -= np.median(mag1)
        q1 = np.percentile(mag1, 25)
        q3 = np.percentile(mag1, 75)
        med = np.percentile(mag1, 50)
        o1 = q1-(q3-q1)
        o3 = q3+(q3-q1)
        print(o1, o3)
        idx = mag > q3
        #mag[np.invert(idx)] = 0
        hsv[:,:,0] = ang*180/np.pi/2
        hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2',rgb)
        cv2.waitKey(1)
        prvs = next