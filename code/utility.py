#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.ndimage.filters as filters

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_dict(dict_):

    """
    Plot a series of subplots where the images are stored in a dictionary.
    Use to plot the dogs.
    """

    ncols = 4
    if len(dict_.keys()) <= ncols:
        ncols = len(dict_.keys())

    nrows = math.ceil(len(dict_.keys())/ncols)
    pos = 1

    for key in dict_.keys():
        plt.subplot(nrows, ncols, pos)
        plt.imshow(dict_[key])
        plt.axis('off')
        pos += 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_keypoints(image, points, show_original = False, save = False, save_name = "keypoints"):

    """
    With an image and list of tuples, plot the points on the image
    """

    if show_original:
        pass #not working or multiple bands
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plt.gray()
        ax1.imshow(image)
        ax1.axis('off')

        for point in points:
            circle = plt.Circle((point[3], point[2]), 3, color='y', fill = False)
            ax2.add_artist(circle)
        ax2.imshow(image)
        ax2.axis('off')

    else:
                
        fig, ax1 = plt.subplots(1, 1)
        
        for point in points:
            circle = plt.Circle((point[3], point[2]), 3, color='y', fill = False)
            ax1.add_artist(circle)
        ax1.imshow(image)
        ax1.axis('off')

    if save:
        save_name = str(save_name)
        fig.savefig(r'../results/' + save_name + ".jpg")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def downsample_image(image, factor):

    """
    Downsample an image to make it smaller.
    Factor = 2 means make half size.
    Factor = 4 means make quater size etc.
    """

    factor = factor//2

    for scale in range(factor):

        downsampled_columns = image[:,range(0,image.shape[1],2)]
        image = downsampled_columns[range(0,image.shape[0],2),:]

    return image.astype(int)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~