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
        image = np.float32(im)/255

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
        W = np.ones(image.shape[:2])

        W_cont = cont ** w_c
        W = np.multiply(W, W_cont) 

        W_sat = sat ** w_s
        W = np.multiply(W, W_sat)

        W_exp = exp ** w_e
        W = np.multiply(W, W_exp) + 1e-12

        wsum = wsum + W
        
        weights.append(W)
        show(W*255,titre='Poids')

    nonzero = wsum > 0
    for i in range(len(weights)):
        weights[i][nonzero]= weights[i][nonzero]/wsum[nonzero]
        
    weights = np.array(weights)
    show(weights,titre='Poids')
    weights = weights.tolist()
        
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

def main(argv):
    path = os.path.dirname(os.path.realpath(__file__)) + "/Images/"
    plt.figure()
    files = [f for f in os.listdir(path)]
    images = []

    for file in files:
        img = cv2.imread(f'Images/{file}')
        images.append(img)

    W = weights_map(images)
    res_naive = naive_fusion(images, W)
    show(res_naive,titre='naive')

if (__name__ == '__main__'):
    main(sys.argv)