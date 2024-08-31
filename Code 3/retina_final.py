# Retipy - Retinal Image Processing on Python
# Copyright (C) 2017-2018  Alejandro Valdes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""retina module to handle basic image processing on retinal images"""

import base64
import math
import numpy as np
import warnings
from copy import copy
from io import BytesIO
#####from function_ import thinning
from matplotlib import pyplot as plt
from os import path
from PIL import Image
from scipy import ndimage
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize
import cv2
import pandas as pd

class Retina(object):
    def __init__(self, vessel: np.ndarray, skeleton: np.ndarray):
        self.np_image=skeleton 
        self.vessel_image=vessel
        self.depth = 1
        self.shape = self.np_image.shape
        
    

class Window(Retina):
    """
    a ROI (Region of Interest) that extends the Retina class
    TODO: Add support for more than depth=1 images (only if needed)
    """
    def __init__(self, image, dimension, method="separated", min_pixels=10):
        super(Window, self).__init__(
            image.np_image,image.vessel_image
            )
        self.windows, self.windows_vessel, self.w_pos = Window.create_windows(image, dimension, method, min_pixels)
       # if len(self.windows) == 0:
        #    raise ValueError("No windows were created for the given retinal image")
        self.shape = self.windows.shape
        #self._mode = self.mode_pytorch
        self._tags = None
        
    def create_windows(
            image, dimension, method="separated", min_pixels=10) -> tuple:
        """
        Creates multiple square windows of the given dimension for the current retinal image.
        Empty windows (i.e. only background) will be ignored

        Separated method will create windows of the given dimension size, that does not share any
        pixel, combined will make windows advancing half of the dimension, sharing some pixels
        between adjacent windows.
        :param image: an instance of Retina, to be divided in windows
        :param dimension:  window size (square of [dimension, dimension] size)
        :param method: method of separation (separated or combined)
        :param min_pixels: ignore windows with less than min_pixels with value.
                           Set to zero to add all windows
        :return: a tuple with its first element as a numpy array with the structure
                 [window, depth, height, width] and its second element as [window, 2, 2]
                 with the window position
        """

        if image.shape[0] % dimension != 0 or image.shape[1] % dimension != 0:
            raise ValueError(
                "image shape is not the same or the dimension value does not divide the image "
                "completely: sx:{} sy:{} dim:{}".format(image.shape[0], image.shape[1], dimension))

        #                      window_count
        windows = []
        windows_position = []
        window_id = 0
        img_dimension = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[1]
        if method == "separated":
            windows = np.empty(
                [(img_dimension // dimension) ** 2, image.depth, dimension, dimension])
            
            windows_vessel = np.empty(
                [(img_dimension // dimension) ** 2, image.depth, dimension, dimension])
            windows_position = np.empty([(img_dimension // dimension) ** 2, 2, 2], dtype=int)
            for x in range(0, image.shape[0], dimension):
                for y in range(0, image.shape[1], dimension):
                    cw = windows_position[window_id]
                    cw[0, 0] = x
                    cw[1, 0] = x + dimension
                    cw[0, 1] = y
                    cw[1, 1] = y + dimension
                    t_window = image.np_image[cw[0, 0]:cw[1, 0], cw[0, 1]:cw[1, 1]]
                    v_window = image.vessel_image[cw[0, 0]:cw[1, 0], cw[0, 1]:cw[1, 1]]
                    if t_window.sum() >= min_pixels:
                        windows[window_id, 0] = t_window
                        windows_vessel[window_id, 0] = v_window
                        window_id += 1
        elif method == "combined":
            new_dimension = dimension // 2
            windows = np.empty(
                [(img_dimension // new_dimension) ** 2, image.depth, dimension, dimension])
            windows_position = np.empty([(img_dimension // new_dimension) ** 2, 2, 2], dtype=int)
            if image.shape[0] % new_dimension != 0:
                raise ValueError(
                    "Dimension value '{}' is not valid, choose a value that its half value can split the image evenly"
                    .format(dimension))
            for x in range(0, image.shape[0] - new_dimension, new_dimension):
                for y in range(0, image.shape[1] - new_dimension, new_dimension):
                    cw = windows_position[window_id]
                    cw[0, 0] = x
                    cw[1, 0] = x + dimension
                    cw[0, 1] = y
                    cw[1, 1] = y + dimension
                    t_window = image.np_image[cw[0, 0]:cw[1, 0], cw[0, 1]:cw[1, 1]]
                    if t_window.sum() >= min_pixels:
                        windows[window_id, 0] = t_window
                        window_id += 1
        if window_id <= windows.shape[0]:
            if window_id == 0:
                windows = []
                windows_vessel=[]
                windows_position = []
            else:
                windows = np.resize(
                    windows, [window_id, windows.shape[1], windows.shape[2], windows.shape[3]])
                windows_vessel = np.resize(
                    windows_vessel, [window_id, windows_vessel.shape[1], windows_vessel.shape[2], windows_vessel.shape[3]])
                windows_position = np.resize(windows_position, [window_id, 2, 2])

        #  print('created ' + str(window_id) + " windows")
        return windows, windows_vessel, windows_position


def detect_vessel_border(image: Retina, ignored_pixels=1):
    """
    Extracts the vessel border of the given image, this method will try to extract all vessel
    borders that does not overlap.

    Returns a list of lists with the points of each vessel.

    :param image: the retinal image to extract its vessels
    :param ignored_pixels: how many pixels will be ignored from borders.
    """

    def neighbours(pixel, window):  # pragma: no cover
        """
        Creates a list of the neighbouring pixels for the given one. It will only
        add to the list if the pixel has value.

        :param pixel: the pixel position to extract its neighbours
        :param window:  the window with the pixels information
        :return: a list of pixels (list of tuples)
        """
        x_less = max(0, pixel[0] - 1)
        y_less = max(0, pixel[1] - 1)
        x_more = min(window.shape[0] - 1, pixel[0] + 1)
        y_more = min(window.shape[1] - 1, pixel[1] + 1)

        active_neighbours = []

        if window.np_image[x_less, y_less] > 0:
            active_neighbours.append([x_less, y_less])
        if window.np_image[x_less, pixel[1]] > 0:
            active_neighbours.append([x_less, pixel[1]])
        if window.np_image[x_less, y_more] > 0:
            active_neighbours.append([x_less, y_more])
        if window.np_image[pixel[0], y_less] > 0:
            active_neighbours.append([pixel[0], y_less])
        if window.np_image[pixel[0], y_more] > 0:
            active_neighbours.append([pixel[0], y_more])
        if window.np_image[x_more, y_less] > 0:
            active_neighbours.append([x_more, y_less])
        if window.np_image[x_more, pixel[1]] > 0:
            active_neighbours.append([x_more, pixel[1]])
        if window.np_image[x_more, y_more] > 0:
            active_neighbours.append([x_more, y_more])

        return active_neighbours
    
    
    def intersection(mask,image, it_x, it_y):
        """
        Remove the intersection in case the whole vessel is too long
        """
        vessel_ = image.np_image
        x_less = max(0, it_x - 1)
        y_less = max(0, it_y - 1)
        x_more = min(vessel_.shape[0] - 1, it_x + 1)
        y_more = min(vessel_.shape[1] - 1, it_y + 1)

        active_neighbours = (vessel_[x_less, y_less]>0).astype('float')+ \
                            (vessel_[x_less, it_y]>0).astype('float')+ \
                            (vessel_[x_less, y_more]>0).astype('float')+ \
                            (vessel_[it_x, y_less]>0).astype('float')+ \
                            (vessel_[it_x, y_more]>0).astype('float')+ \
                            (vessel_[x_more, y_less]>0).astype('float')+ \
                            (vessel_[x_more, it_y]>0).astype('float')+ \
                            (vessel_[x_more, y_more]>0).astype('float')

        if active_neighbours > 2:
            cv2.circle(mask,(it_y,it_x),radius=1,color=(0,0,0),thickness=-1)
        

        return mask,active_neighbours        
        
    '''
    # original remove x duplicate
    def vessel_extractor(window, start_x, start_y):
        """
        Extracts a vessel using adjacent points, when each point is extracted is deleted from the
        original image
        & Measure width
        """
        vessel = []
        width_list = []
        width_mask = np.zeros((window.np_image.shape))
        pending_pixels = [[start_x, start_y]]
        while pending_pixels:
            pixel = pending_pixels.pop(0)
            if window.np_image[pixel[0], pixel[1]] > 0:
                vessel.append(pixel)
                window.np_image[pixel[0], pixel[1]] = 0

                # add the neighbours with value to pending list:
                pending_pixels.extend(neighbours(pixel, window))

        # sort by x position
        vessel.sort(key=lambda item: item[0])

        # remove all repeating x values???????????
        current_x = -1
        filtered_vessel = []
        for pixel in vessel:
            if pixel[0] == current_x:
                pass
            else:
                filtered_vessel.append(pixel)
                current_x = pixel[0]

        vessel_x = []
        vessel_y = []
        for pixel in filtered_vessel:
            vessel_x.append(pixel[0])
            vessel_y.append(pixel[1])
            

        return [vessel_x, vessel_y]
    '''
    
    # 2021/10/31 remove setting of the sort & x duplication
    def vessel_extractor(window, start_x, start_y):
        """
        Extracts a vessel using adjacent points, when each point is extracted is deleted from the
        original image
        & Measure width
        """
        vessel = []
        width_list = []
        width_mask = np.zeros((window.np_image.shape))
        pending_pixels = [[start_x, start_y]]
        while pending_pixels:
            pixel = pending_pixels.pop(0)
            if window.np_image[pixel[0], pixel[1]] > 0:
                vessel.append(pixel)
                window.np_image[pixel[0], pixel[1]] = 0

                # add the neighbours with value to pending list:
                pending_pixels.extend(neighbours(pixel, window))

        # sort by x position
        '''
        vessel.sort(key=lambda item: item[0])

        # remove all repeating x values???????????
        current_x = -1
        filtered_vessel = []
        for pixel in vessel:
            if pixel[0] == current_x:
                pass
            else:
                filtered_vessel.append(pixel)
                current_x = pixel[0]
        '''
        filtered_vessel = vessel
        vessel_x = []
        vessel_y = []
        for pixel in filtered_vessel:
            vessel_x.append(pixel[0])
            vessel_y.append(pixel[1])
            

        return [vessel_x, vessel_y]
    
    
    vessels = []
    active_neighbours_list = []
    width_list_all = []
    mask_ = np.ones((image.np_image.shape))
    
    for it_x in range(ignored_pixels, image.shape[0] - ignored_pixels):
        for it_y in range(ignored_pixels, image.shape[1] - ignored_pixels):
            if image.np_image[it_x, it_y] > 0:
                mask,active_neighbours = intersection(mask_,image, it_x, it_y)
                active_neighbours_list.append(active_neighbours)
    ##plt.imshow(image.np_image)
    image.np_image = image.np_image * mask
    ##plt.imshow(image.np_image)
    #cv2.imwrite('./intersection_test/{}.png'.format(image._file_name),image.np_image)
    
    for it_x in range(ignored_pixels, image.shape[0] - ignored_pixels):
        for it_y in range(ignored_pixels, image.shape[1] - ignored_pixels):
            if image.np_image[it_x, it_y] > 0:
                vessel = vessel_extractor(image, it_x, it_y)
                vessels.append(vessel)
    
                
    return vessels