import glob
import matplotlib.pyplot as plt
import sys
import os


paths = glob.glob(sys.argv[1])

for f in paths:
    img = plt.imread(f)
    if(img.shape != (256,256,3)):
        print(f," has shape ",img.shape)
        os.remove(f)
