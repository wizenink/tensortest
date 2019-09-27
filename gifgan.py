import imageio
import glob
import cv2

image_path = 'images/*.png'

files = glob.glob(image_path)
images = []
for f in files:
    image = imageio.imread(f)
    images.append(image)

imageio.mimsave("progress.gif",images)