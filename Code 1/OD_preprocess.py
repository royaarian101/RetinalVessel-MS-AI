import cv2
import numpy as np
from skimage.exposure import match_histograms 
from skimage.morphology import (erosion, dilation, closing, opening)
def pre_processing(images,img_reference,images_name,save_path1,kernel):
    
    img_process = []
    img_invert,img_match = [],[]
   # kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    for index, item in enumerate(images_name):
        if item==img_reference+'.png':
            index_ref=index
            break
    im_inverted_ref=cv2.bitwise_not(images[index_ref])        
    for i in range(len(images)):
        #inverting image
        im_inverted=cv2.bitwise_not(images[i])
        ###matching image with ref image
        im_matched=np.uint8(match_histograms(im_inverted,im_inverted_ref))
        ###removing central reflection of blood vessels 
        im_open=opening(im_matched.copy(),kernel)
        im=closing(im_open.copy(),kernel)
        image=cv2.cvtColor(np.uint8(im),cv2.COLOR_GRAY2RGB)
        img_process.append(image)
        img_invert.append(im_inverted)
        img_match.append(im_matched)
        cv2.imwrite(save_path1+images_name[i], image)        
    return index_ref,img_invert,img_match,img_process
