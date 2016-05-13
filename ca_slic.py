import os
PI = 3.1415
base_path = os.getcwd()+'/'
SHOW_IMG = True

import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.cm as cm
import time
from PIL import Image
import sys
import math
import pdb
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_super_background_indexs(image, quantile=0.35, ignored_indexs=None):
    # not a gray scale
    if ignored_indexs is None:
        ignored_indexs = []

    image = np.asarray(image)
    if len(image.shape)>2 and image.shape[2] > 1:
        image = rgb2gray(image)

    image_flat = image.flatten()
    # set as maximum to be ignored
    image_flat[ignored_indexs] = 2.0
    indexs = np.argsort(image_flat)[:quantile*image_flat.size]
    
    return indexs

def get_background_indexs(image, output_image_path, quantile=0.35, ignored_indexs=None):
    # not a gray scale
    if ignored_indexs is None:
        ignored_indexs = []
    if len(image.shape)>2 and image.shape[2] > 1:
        image = rgb2gray(image) 
    
    image_flat = image.flatten()
    # set as maximum to be ignored
    image_flat[ignored_indexs] = 2.0
    indexs = np.argsort(image_flat)[:quantile*image_flat.size]
    
    background_image_flat = np.ones((image.shape[0]*image.shape[1]))
    background_image_flat[indexs] = 0
    background_image = background_image_flat.reshape((image.shape[0], image.shape[1]))
    output_image_PIL = Image.fromarray((background_image*255.).astype(np.uint8))
    output_image_path = output_image_path[:-4]+'-background'+output_image_path[-4:]
    output_image_PIL.save(output_image_path)
    
    return indexs

def get_foreground_indexs(image, output_image_path, quantile=0.05, ignored_indexs=None):
    if ignored_indexs is None:
        ignored_indexs = []
    # not a gray scale
    if len(image.shape)>2 and image.shape[2] > 1:
        image = rgb2gray(image) 
    
    image_flat = image.flatten()
    image_flat[ignored_indexs] = -1.0
    indexs = np.argsort(image_flat,)[-quantile*image_flat.size:]

    foreground_image_flat = np.ones((image.shape[0]*image.shape[1]))
    foreground_image_flat[indexs] = 0
    foreground_image = foreground_image_flat.reshape((image.shape[0], image.shape[1]))
    output_image_PIL = Image.fromarray((foreground_image*255.).astype(np.uint8))
    output_image_path = output_image_path[:-4]+'-foreground'+output_image_path[-4:]
    output_image_PIL.save(output_image_path)
    
    return indexs

def get_super_foreground_indexs(image, quantile=0.05, ignored_indexs=None):
    if ignored_indexs is None:
        ignored_indexs = []
    # not a gray scale
    image = np.asarray(image)
    if len(image.shape)>2 and image.shape[2] > 1:
        image = rgb2gray(image) 
    
    image_flat = image.flatten()
    image_flat[ignored_indexs] = -1.0
    indexs = np.argsort(image_flat,)[-quantile*image_flat.size:]

    return indexs

def unique_append(l, e):
    if e not in l:
        l.append(e)

def get_superpixel(image, num_segments=500):
    """
    Args:
        image(N*M array):
    
    Returns:
        labels(N*M array), new_neighbors(n*[int]), new_rgb(n*[int, int, int])
    """

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    (N, M, _) = image.shape

    labels = slic(image, n_segments = num_segments, sigma = 5)
    
    # Show segmentation
    # fig = plt.figure("Superpixels -- %d segments" % (num_segments))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(image, labels))
    # plt.axis("off")
    # plt.show()

    n_labels = np.max(labels)+1

    new_neighbors = [[] for i in xrange(n_labels)]
    new_rgb = np.zeros((n_labels, 3))
    label_count = n_labels*[0]
    
    for i in xrange(N):
        for j in xrange(M):
            if j < M-1 and labels[i, j] != labels[i, j+1]:
                unique_append(new_neighbors[labels[i, j]], labels[i, j+1])
                unique_append(new_neighbors[labels[i, j+1]], labels[i, j])
            if i < N-1 and j < M-1 and labels[i, j] != labels[i+1, j+1]:
                unique_append(new_neighbors[labels[i, j]], labels[i+1, j+1])
                unique_append(new_neighbors[labels[i+1, j+1]], labels[i, j])
            if i < N-1 and labels[i, j] != labels[i+1, j]:
                unique_append(new_neighbors[labels[i, j]], labels[i+1, j])
                unique_append(new_neighbors[labels[i+1, j]], labels[i, j])
            new_rgb[labels[i, j],:] += image[i,j,:]
            label_count[labels[i, j]] += 1

    for i in xrange(n_labels):
        new_rgb[i] /= label_count[i]

    return labels, new_neighbors, new_rgb

def get_supersaliency(labels, saliency_image):
    if len(saliency_image.shape)>2 and saliency_image.shape[2] > 1:
        saliency_image = rgb2gray(saliency_image) 

    labels_flatten = labels.flatten()
    saliency_flatten = saliency_image.flatten()

    n_labels = np.max(labels)+1
    super_saliency = n_labels*[0]
    label_count = n_labels*[0]

    for i in xrange(n_labels):
        indexs = np.argwhere(labels_flatten==i)
        super_saliency[i] = np.mean(saliency_flatten[indexs])

    return super_saliency


def get_super_index(labels, foreground_indexs, background_indexs):
    """
    Args:
        labels(N*M array):

        foreground_indexs([int]):

        background_indexs([int]):

    Returns:
        super_foreground_indexs([int]), super_background_indexs([int])

    """
    labels_flatten = labels.flatten()
    n_labels = np.max(labels)+1
    super_background_indexs = []
    super_foreground_indexs = []

    for index in background_indexs:
        unique_append(super_background_indexs, labels_flatten[index])
    for index in foreground_indexs:
        unique_append(super_foreground_indexs, labels_flatten[index])

    # pdb.set_trace()
    # remove overlap from background
    super_background_indexs = [index for index in super_background_indexs if index not in super_foreground_indexs]

    return super_foreground_indexs, super_background_indexs

def get_salience_indexs(saliency, threshold=0.75):
    saliency_flatten = saliency.flatten()
    return [i for i in xrange(len(saliency_flatten)) if saliency_flatten[i] > threshold]

def get_fg_bg(labels, fg_indexs, bg_indexs):
    labels_flatten = labels.flatten()
    image_flatten = 0.5*np.ones(labels_flatten.shape)

    for index in fg_indexs:
        image_flatten[np.argwhere(labels_flatten==index)] = 1.
    for index in bg_indexs:
        image_flatten[np.argwhere(labels_flatten==index)] = 0.

    return image_flatten.reshape(labels.shape)

def ca(neighbors, rgbs, fg_indexs, bg_indexs, sigma_3_square=0.1, a=0.6, b=0.2, num_step=10, fg_bias=0.3, bg_bias=-0.3):
    """

    Returns:
        super_saliency()
    """
    N = len(neighbors)

    F = np.asmatrix(np.zeros((N, N)))
    f = lambda c1, c2: np.exp(-LA.norm(c1-c2)/(sigma_3_square))
    scale_to_1 = lambda x: (x-np.min(x))/(np.max(x)-np.min(x)) if (np.max(x)-np.min(x))!=0 else 0
    other_indexs = [i for i in range(N) if (i not in fg_indexs and i not in bg_indexs)]
    
    for i in xrange(len(neighbors)):
        for j in neighbors[i]:
            F[j, i] = F[i, j] = f(rgbs[i,:], rgbs[j, :])
        D = np.asmatrix(np.diag(np.asarray(np.sum(F, axis=1)).flatten()))

    F_star = inv(D)*F

    C_diag = np.asarray(np.divide(1, np.max(F, axis=1))).flatten()
    C_max, C_min = np.max(C_diag), np.min(C_diag)
    C_star_diag = a*(C_diag-C_min)/(C_max-C_min)+b
    C_star = np.asmatrix(np.diag(C_star_diag))
    
    S_0 = 0.5*np.ones((N, 1))
    S_0 = np.asmatrix(S_0)
    S_new = S = S_0

    func = lambda x: 1. if x > 1. else (0. if x < 0. else x)
    
    for i in xrange(num_step):
        S_new[fg_indexs] += fg_bias
        S_new[bg_indexs] += bg_bias
    
        S_new = C_star*S_new + (np.identity(N)-C_star)*F_star*S_new

        if len(other_indexs) > 0:
            S_new[other_indexs] = scale_to_1(S_new[other_indexs])
        
        S_new = np.vectorize(func)(S_new)

        print i, "iteration"
        print "Norm of difference:", LA.norm(S_new-S)
        S = S_new

    for i in xrange(num_step):    
        S_new = C_star*S + (np.identity(N)-C_star)*F_star*S
        S_new = scale_to_1(S_new)
        print i, "iteration"
        print "Norm of difference:", LA.norm(S_new-S)
        S = S_new

    return S

def ca_multilabel(neighbors, rgbs, super_saliency_list, fg_indexs_list, bg_indexs_list, sigma_3_square=0.1, a=0.6, b=0.2, num_step=10, fg_bias=0.3, bg_bias=-0.3):
    """

    Returns:
        super_saliency()
    """
    if(len(fg_indexs_list) != len(bg_indexs_list)):
        raise Exception("The dimension of background and foreground should be the same")

    n_labels = len(fg_indexs_list)

    N = len(neighbors)

    F = np.zeros((N, N))
    f = lambda c1, c2: np.exp(-LA.norm(c1-c2)/(sigma_3_square))
    func = lambda x: 1. if x > 1. else (0. if x < 0. else x)
    scale_to_1 = lambda x: (x-np.min(x))/(np.max(x)-np.min(x)) if (np.max(x)-np.min(x))!=0 else 0

    # other_indexs = [i for i in range(N) if (i not in fg_indexs and i not in bg_indexs)]
    
    for i in xrange(len(neighbors)):
        for j in neighbors[i]:
            F[j, i] = F[i, j] = f(rgbs[i,:], rgbs[j, :])
        D = np.diag(np.asarray(np.sum(F, axis=1)).flatten())

    F_star = inv(D).dot(F)

    C_diag = np.asarray(np.divide(1, np.max(F, axis=1))).flatten()
    C_max, C_min = np.max(C_diag), np.min(C_diag)
    C_star_diag = a*(C_diag-C_min)/(C_max-C_min)+b
    C_star = np.diag(C_star_diag)
    
    # S_0 = 0.5*np.ones((N, n_labels+1))

    S_0 = np.asarray(super_saliency_list).transpose((1,0))
    for j in xrange(n_labels):
        S_0[:, j] = scale_to_1(S_0[:, j])
    S_0 = np.append(S_0, 1-np.max(S_0, axis=1)[:, np.newaxis], axis=1)

    S_new = S = S_0
    
    for i in xrange(num_step):
        for j in xrange(n_labels):
            S_new[fg_indexs_list[j], j] += fg_bias
            # S[bg_indexs_list[j], j] += bg_bias

        S_new = C_star.dot(S_new) + (np.identity(N)-C_star).dot(F_star).dot(S_new)

        # TODO
        # if len(other_indexs) > 0:
        #     S_new[other_indexs] = scale_to_1(S_new[other_indexs])
        # S_new = np.vectorize(func)(S_new)

        S_new = scale_to_1(S_new)
        S_new = (S_new-0.5)*math.pi
        S_new = np.tan(S_new)
        # avoid overflow

        S_new = S_new - np.max(S_new, axis=1)[:, np.newaxis]
        S_new = np.exp(S_new)
        S_new = np.divide(S_new+0.0000000001, np.sum(S_new+0.0000000001, axis=1)[:, np.newaxis])

        print i, "iteration"
        print "Norm of difference:", LA.norm(S_new-S)
        S = S_new

    for i in xrange(num_step):    
        S_new = C_star.dot(S) + (np.identity(N)-C_star).dot(F_star).dot(S)
        S_new = scale_to_1(S_new)
        S_new = (S_new-0.5)*math.pi
        S_new = np.tan(S_new)
        # avoid overflow

        S_new = S_new - np.max(S_new, axis=1)[:, np.newaxis]
        S_new = np.exp(S_new)
        S_new = np.divide(S_new+0.0000000001, np.sum(S_new+0.0000000001, axis=1)[:, np.newaxis])

        print i, "iteration"
        print "Norm of difference:", LA.norm(S_new-S)
        S = S_new

    return S

def get_saliency(labels, super_saliency):
    n_labels = len(super_saliency)
    
    labels_flatten = labels.flatten()
    saliency_flatten = np.zeros(labels_flatten.shape)

    for i in xrange(n_labels):
        indexs = np.argwhere(labels_flatten==i)
        saliency_flatten[indexs] = super_saliency[i]

    return saliency_flatten.reshape(labels.shape)

def cut_saliency(image, salience_indexs):
    flatten = image.reshape(image.shape[0]*image.shape[1],3)
    flatten[salience_indexs, :] = 0
    return flatten.reshape(image.shape)

def main():
    '''
    Multi-label saliency refinement
    
    usage: python ca_slic.py [-h] [-i IMAGE] [-sl SALIENCY_LIST [SALIENCY_LIST ...]]
               [-rsl OUTPUT_SALIENCY_LIST [OUTPUT_SALIENCY_LIST ...]]
               [-ns NUM_SEGMENTS] [-fq FG_QUANTILE] [-bq BG_QUANTILE]
               [-fb FG_BIAS] [-bb BG_BIAS] [-d]

    optional arguments:
      -h, --help            show this help message and exit
      -i IMAGE, --image IMAGE
                            Image file
      -sl SALIENCY_LIST [SALIENCY_LIST ...], --saliency_list SALIENCY_LIST [SALIENCY_LIST ...]
                            Saliency map file list
      -rsl OUTPUT_SALIENCY_LIST [OUTPUT_SALIENCY_LIST ...], --output_saliency_list OUTPUT_SALIENCY_LIST [OUTPUT_SALIENCY_LIST ...]
                            Refined saliency map file (output)
      -ns NUM_SEGMENTS, --num_segments NUM_SEGMENTS
                            Number of segments of superpixel
      -fql FG_QUANTILE_LIST, --fg_quantile FG_QUANTILE
                            Foreground quantile list
      -bq BG_QUANTILE, --bg_quantile BG_QUANTILE
                            Background quantile
      -fb FG_BIAS, --fg_bias FG_BIAS
                            Foreground bias
      -bb BG_BIAS, --bg_bias BG_BIAS
                            Background bias
      -d, -debug            Save intermediate image for debugging
    '''
    if len(sys.argv) < 3:
        raise ValueError("Please input the image file and saliency file")

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image', type=str, help="Image file")
    parser.add_argument('-sl', '--saliency_list', nargs='+', type=str, help="Saliency map file list")
    parser.add_argument('-rsl', '--output_saliency_list', nargs='+', type=str, help="Refined saliency map file (output)")
    parser.add_argument('-ns', '--n_segments', type=int, help="Number of segments of superpixel")
    parser.add_argument('-fql', '--fg_quantile_list', nargs='+', type=float, help="Foreground quantile list")
    parser.add_argument('-bq', '--bg_quantile', type=float, help="Background quantile")
    parser.add_argument('-fb', '--fg_bias', type=float, help="Foreground bias")
    parser.add_argument('-bb', '--bg_bias', type=float, help="Background bias")
    parser.add_argument('-d', '--debug', action='store_true', help="Save intermediate image for debugging")

    n_segments = parser.parse_args().n_segments
    fg_quantile_list = parser.parse_args().fg_quantile_list
    bg_quantile = parser.parse_args().bg_quantile
    fg_bias = parser.parse_args().fg_bias
    bg_bias = parser.parse_args().bg_bias
    SAVE_INTERMEDIATE_IMG = parser.parse_args().debug
    SHOW_IMG = False
    
    input_image_path = base_path+parser.parse_args().image
    saliency_image_path_list = [base_path+saliency_image 
                                 for saliency_image in parser.parse_args().saliency_list]
    output_saliency_list = [base_path+output_saliency 
                             for output_saliency in parser.parse_args().output_saliency_list]

    n_labels = len(saliency_image_path_list)

    image = img_as_float(io.imread(input_image_path))
    saliency_image_list = [img_as_float(io.imread(saliency_image_path)) 
                            for saliency_image_path in saliency_image_path_list]
    # image = img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))

    labels, neighbors, rgbs = get_superpixel(image, num_segments=n_segments)
    super_saliency_list = [get_supersaliency(labels, saliency_image)
                            for saliency_image in saliency_image_list]
    saliency_list = [get_saliency(labels, super_saliency)
                      for super_saliency in super_saliency_list]
    
    if SAVE_INTERMEDIATE_IMG:
        for i in xrange(n_labels):
            io.imsave(saliency_image_path_list[i][:-4]+'_saliency.png', saliency_list[i])
    if SHOW_IMG:
        for i in xrange(n_labels):
            plt.imshow(saliency_list[i], cmap=plt.get_cmap('gray'))
            plt.show()

    # super_foreground_indexs, super_background_indexs = get_super_index(labels, foreground_indexs, background_indexs)

    super_foreground_indexs_list = [get_super_foreground_indexs(super_saliency_list[i], quantile=fg_quantile_list[i])
                                     for i in xrange(n_labels)]
    super_background_indexs_list = [get_super_background_indexs(super_saliency, quantile=bg_quantile)
                                     for super_saliency in super_saliency_list]
    fg_bg_image_list = [get_fg_bg(labels, super_foreground_indexs_list[i], super_background_indexs_list[i])
                         for i in xrange(len(saliency_image_path_list))]

    if SAVE_INTERMEDIATE_IMG:
        for i in xrange(n_labels):
            io.imsave(saliency_image_path_list[i][:-4]+'_fg_bg_image.png', fg_bg_image_list[i])
    if SHOW_IMG:
        for i in xrange(n_labels):
            plt.imshow(fg_bg_image_list[i], cmap=plt.get_cmap('gray'))
            plt.show()

    super_refined_saliency_array = ca_multilabel(neighbors, rgbs, super_saliency_list, super_foreground_indexs_list, super_background_indexs_list, fg_bias=fg_bias, bg_bias=bg_bias)

    refined_saliency_list = []
    for i in xrange(n_labels+1):
        refined_saliency_list.append(get_saliency(labels, super_refined_saliency_array[:, i]))

    if SHOW_IMG:
        for i in xrange(n_labels+1):
            plt.imshow(refined_saliency_list[i], cmap=plt.get_cmap('gray'))
            plt.show()

    for i in xrange(n_labels):
        io.imsave(output_saliency_list[i], refined_saliency_list[i])

    io.imsave(input_image_path[:-4]+'_background_saliency.png', refined_saliency_list[-1])

if __name__ == '__main__':
    main()