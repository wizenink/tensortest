import cv2
import numpy as np
import os


def mapFiles(rootpath,extension=".jpg"):
    for dirpath,_,fnames in os.walk(rootpath):
        for f in fnames:
            if f.endswith(extension):
                fpath = os.path.join(dirpath,f)
                img = cv2.imread(fpath)
                imgYUV = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
                cv2.imwrite(fpath,imgYUV)



mapFiles("/home/mads/davidmaseda/.keras/datasets/lin/train")