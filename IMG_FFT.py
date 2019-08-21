# -*- coding: utf-8 -*-
# Ref:
# Org: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
# JPN: http://lang.sist.chukyo-u.ac.jp/classes/OpenCV/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html

import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('data/Canon_cornfield.png',0)
f = np.fft.fft2(img1)
fshift = np.fft.fftshift(f)
mag1 = 20*np.log(np.abs(fshift))

img2 = cv2.imread('data/Canon_pens.png',0)
f = np.fft.fft2(img2)
fshift = np.fft.fftshift(f)
mag2 = 20*np.log(np.abs(fshift))


fig, ((axLU, axRU), (axLL, axRL)) = plt.subplots(nrows=2, ncols=2, figsize=(10,6))

axLU.imshow(img1, cmap = 'gray')
axLU.set_title('Corn Field'),
axLU.set_xticks([])
axLU.set_yticks([]) # Remove ticks

axRU.imshow(mag1, cmap = 'gray')
axRU.set_title('Magnitude Spectrum')
axRU.set_xticks([])
axRU.set_yticks([])

axLL.imshow(img2, cmap = 'gray')
axLL.set_title('Pens'),
axLL.set_xticks([])
axLL.set_yticks([]) # Remove ticks

axRL.imshow(mag2, cmap = 'gray')
axRL.set_title('Magnitude Spectrum')
axRL.set_xticks([])
axRL.set_yticks([])


plt.show()
