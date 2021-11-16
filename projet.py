import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from PIL import Image

def show(im,titre='none'):
    if im.shape[0]==3:
        imnew=im.transpose(1,2,0)
    else:
        imnew=im

    cv2.imshow('image', imnew)
    cv2.waitKey(10000)

def weights_map(images):
    (w_c, w_s, w_e) = (1, 1, 1)

    weights = []
    wsum = np.zeros(images[0].shape[:2], dtype=np.float32)

    i = 0

    for im in images:
        image = im/255

        # contrast
        src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(src_gray, cv2.CV_32F)
        cont = np.absolute(laplacian)
                
        # saturation
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        hue,sat,val = cv2.split(hsv)

        # well-exposedness
        img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        red, green, blue = cv2.split(img)
        sigma = 0.2
        red_exp = np.exp(-(red - 0.5)**2 / (2 * sigma**2))
        green_exp = np.exp(-(green - 0.5)**2 / (2 * sigma**2))
        blue_exp = np.exp(-(blue - 0.5)**2 / (2 * sigma**2))
        exp = red_exp * green_exp * blue_exp
        
        #Weights
        W = np.ones(image.shape[:2], dtype=np.float32)

        W = (cont ** w_c) * (sat ** w_s) * (exp ** w_e) + 1e-12
        
        wsum = wsum + W
        
        weights.append(W)

        # show(W*255,titre='Poids')

    nonzero = wsum > 0
    for i in range(len(weights)):
        weights[i][nonzero]= weights[i][nonzero]/wsum[nonzero]
        # weights[i] = weights[i]*255
        
    # weights = np.array(weights)
#   show(weights,titre='Poids')
    # weights = weights.tolist()
        
    return weights
       
def naive_fusion(images, weights):
    zeros = np.zeros(images[0].shape[:2])
    pil_image = Image.fromarray(zeros, 'RGB')
    open_cv_image = np.array(pil_image)
    naive_image = open_cv_image[:, :, ::-1].copy() 
    for channel in range(3):
        for i in range(len(images)):
            naive_image[:, :, channel] = naive_image[:, :, channel] + (weights[i] * images[i][:, :, channel])
    return naive_image

def reduce(image):
    kernel = cv2.getGaussianKernel(ksize=4, sigma=0.2) #Gaussian filter coefficients   
    reduced = cv2.filter2D(image, -1, kernel) #Convolution of the image with the kernel
    reduced = cv2.resize(reduced, None, fx=0.5, fy=0.5)
    return reduced

def gaussian_pyramid(image, size):
    img = image.copy()
    gaussian = [img]
    for i in range(size):
        img = reduce(img)
        gaussian.append(img)
    return gaussian

def expand(image):
    kernel = cv2.getGaussianKernel(ksize=4, sigma=0.2)
    expanded = cv2.resize(image, None, fx=2, fy=2)
    expanded = cv2.filter2D(expanded, -1, kernel)
    expanded = np.float32(expanded)
    return expanded

def laplacian_pyramid(image, size):
    gaussian = gaussian_pyramid(image, size+1)
    laplacian = [gaussian[size-1]]
    for i in range(size-1, 0, -1):
        exp = expand(gaussian[i])
        l = cv2.subtract(gaussian[i-1], exp)
        laplacian = [l] + laplacian
    return laplacian

def collapse(pyramid):
    size = len(pyramid)
    collapsed = pyramid[size-1]
    for i in range(size-2, -1, -1):
        collapsed = cv2.add(expand(collapsed), pyramid[i])
    return collapsed

def fusion(images, weights, size):
    l = []
    g = []
    for (image, weight) in zip(images, weights):
        g.append(gaussian_pyramid(weight, size))
        l.append(laplacian_pyramid(image, size))
        
    pyramid_weigth = []
    for i in range(size):
        pw = np.zeros(l[0][i].shape, dtype=np.float32)
        for k in range(len(images)):
            lp = l[k][i]
            gps = np.float32(g[k][i])/255
            gp = np.dstack((gps, gps, gps)) #Stack arrays in sequence depth wise
            lp_gp = cv2.multiply(lp, gp, dtype=cv2.CV_32F)
            pw = cv2.add(pw, lp_gp)
            
        pyramid_weigth.append(pw)

    fusion = collapse(pyramid_weigth)
    return fusion


def main(argv):
    path = os.path.dirname(os.path.realpath(__file__)) + "/Images/"
    files = [f for f in os.listdir(path)]
    images = []

    for file in files:
        img = cv2.imread(f'Images/{file}')
        images.append(img.astype(np.float32))

    W = weights_map(images) #Weigths

    res_naive = naive_fusion(images, W) #Naive Fusion
    # show(res_naive,titre='naive')
    res = fusion(images, W, 3)
    show(res)


if (__name__ == '__main__'):
    main(sys.argv)