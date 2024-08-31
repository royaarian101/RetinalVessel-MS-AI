import os
import cv2
from matplotlib import pyplot as plt

def load_images_from_folder(folder):
    images = []
    images_name=[]
    for filename in os.listdir(folder):
        images_name.append(filename)
        img = cv2.imread(os.path.join(folder,filename))        
        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img=cv2.resize(gray_img,(512,512))
            #normalize:
            gray_img=cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
            images.append(gray_img)
    return images, images_name

