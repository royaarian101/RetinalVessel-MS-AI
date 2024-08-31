import cv2
import numpy as np
from skimage.exposure import match_histograms
def pre_process_vessel(images,img_ref,images_name,path_pre_process):
   
    for index, item in enumerate(images_name):
        if item==img_ref+'.png':
            index_ref=index
            break
                    
    img_process = []       
    for i in range(len(images)):
        im_matched=np.uint8(match_histograms(images[i],images[index_ref]))
        image=cv2.cvtColor(np.uint8(im_matched),cv2.COLOR_GRAY2RGB)
        img_process.append(image)
        cv2.imwrite(path_pre_process+images_name[i], image)        
    return index_ref,img_process