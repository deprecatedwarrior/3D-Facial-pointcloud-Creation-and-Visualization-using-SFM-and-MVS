import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/Users/avinashkaur/Desktop/MS-EE/Computer Vision CSCI677/Homeworks/Homework2/BSD IMAGES/sculpture.jpg')
edges = cv2.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
