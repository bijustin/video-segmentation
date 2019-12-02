""" 
Unsupervised Online Video Object Segmentation
with Motion Property Understanding

paper link: https://arxiv.org/pdf/1810.03783.pdf
github repo: https://github.com/visiontao/uovos
"""

import cv2
import numpy as np


class Baseline():

    def __init__(self, filename, Th=10, Fth=1):
        self.cap = cv2.VideoCapture("../videos/" + filename)
        self.Th = Th
        self.Fth = Fth

        ret, self.previous_I = self.cap.read()
        self.previous_I = cv2.cvtColor(self.previous_I, cv2.COLOR_BGR2GRAY)
        self.x = np.size(self.previous_I, axis=0)
        self.y = np.size(self.previous_I, axis=1)

        self.current_I = None
        self.current_SI = np.zeros((self.x, self.y))
        self.current_BG = np.zeros((self.x, self.y))
        self.current_BI = np.zeros((self.x, self.y))
        self.current_BD = np.zeros((self.x, self.y))
        self.current_FD = np.zeros((self.x, self.y))
        self.current_FDM = np.zeros((self.x, self.y))
        self.current_BDM = np.zeros((self.x, self.y))
        self.current_IOM = np.zeros((self.x, self.y))
        
        self.previous_IOM = np.zeros((self.x, self.y))
        self.previous_BDM = np.zeros((self.x, self.y))    
        self.previous_SI = np.zeros((self.x, self.y))
        self.previous_BG = np.zeros((self.x, self.y))
        self.previous_BI = np.zeros((self.x, self.y))
        self.previous_BD = np.zeros((self.x, self.y))
        self.previous_FD = np.zeros((self.x, self.y))
        self.previous_FDM = np.zeros((self.x, self.y))

    def __frameDifference(self):
        self.current_FD = np.absolute(self.current_I - self.previous_I)
        self.current_FDM = np.where(self.current_FD < self.Th, 0, 1)

    def __backgroundRegistration(self):
        self.current_SI = self.previous_SI
        
        for x in range(self.x):
            for y in range(self.y):
                if self.current_FDM[x, y] == 0:
                    self.current_SI[x, y] = self.previous_SI.item((x, y)) + 1
                else:
                    self.current_SI[x, y] = 0
        
        for x in range(self.x):
            for y in range(self.y):
                if self.current_SI.item((x, y)) == self.Fth:
                    self.current_BG[x, y] = self.current_I[x, y]
                else:
                    self.current_BG[x, y] = self.previous_BG[x, y]
                
        for x in range(self.x):
            for y in range(self.y):
                if self.current_SI.item((x, y)) == self.Fth:
                    self.current_BI[x, y] = 1
                else:
                    self.current_BI[x, y] = self.previous_BI[x, y]
        
    
    def __backgroundDifferenceMask(self):
        self.current_BD = np.absolute(self.current_I - self.previous_BG)

        self.current_BDM = np.where(self.current_BD < self.Th, 0, 1)

    def __objectDetection(self):
         for x in range(self.x):
            for y in range(self.y):
                if self.current_BI[x, y] == 1:
                    self.current_IOM[x, y] = self.current_BDM[x, y]
                else:
                    self.current_IOM[x, y] = self.current_FDM[x, y]

    def __postProcessing(self):
         for x in range(self.x):
            for y in range(self.y):
                if self.current_IOM[x, y] == 1:
                    self.current_SI[x, y] = 0

    def __updateParam(self):
        self.previous_I = self.current_I
        self.previous_SI = self.current_SI
        self.previous_BG = self.current_BG
        self.previous_BI = self.current_BI
        self.previous_BD = self.current_BD
        self.previous_FD = self.current_FD
        self.previous_FDM = self.current_FDM
        self.previous_BDM = self.current_BDM
        self.previous_IOM = self.current_IOM

    def propagate(self):
        while True:
            ret, self.current_I = self.cap.read()
            self.current_I = cv2.cvtColor(self.current_I, cv2.COLOR_BGR2GRAY)
            if not ret:
                break

            self.__frameDifference()
            self.__backgroundRegistration()
            self.__backgroundDifferenceMask()
            self.__objectDetection()
            self.__postProcessing()
            self.__updateParam()

            self.current_IOM *= 255
            self.current_IOM = self.current_IOM.astype(np.uint8)
            print(self.current_IOM)
            
            cv2.imshow('frame2', self.current_IOM)
            cv2.waitKey(1)



if __name__ == "__main__":
    model = Baseline("bus.mp4")  
    model.propagate()     
