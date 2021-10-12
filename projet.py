import sys  

import numpy as np
import matplotlib.pyplot as plt
import cv2

def weights_map(images):
    (w_c, w_s, w_e) = (1, 1, 1)

    weights = []
    wsum = np.zeros(images[0].shape[:2]) #size of the image

    i = 0

    for im in images:
        image = np.float32(im)/255 #normalizes each image

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
        W = np.multiply(W, W_exp)
    
        wsum = wsum + W

        weights.append(W)

    nonzero = wsum > 0 #normalizes weigth
    for i in range(len(weights)):
        weights[i][nonzero] = weights[i][nonzero]/sum[nonzero]
        weights[i] = np.uint8(weights[i]*255)

    return weights
       

def main(argv):
    images = cv2.imread('figA.png')


if (__name__ == '__main__'):
    main(sys.argv)