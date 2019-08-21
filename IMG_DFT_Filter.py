# -*- coding: utf-8 -*-
# Ref:
# Org: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
# JPN: http://lang.sist.chukyo-u.ac.jp/classes/OpenCV/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('data/baboon.jpg',0)

dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

rows, cols = img.shape
crow,ccol = int(rows/2) , int(cols/2)

isize=10 # マスクの半分のサイズ

# ローパスフィルタ用マスクの作成，中心正方領域は1，それ以外は0
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-isize:crow+isize, ccol-isize:ccol+isize] = 1
# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back1 = cv2.idft(f_ishift)
img_back1 = cv2.magnitude(img_back1[:,:,0],img_back1[:,:,1])

# ハイパスフィルタ用マスクの作成，中心正方領域は0，それ以外は1
mask = np.ones((rows,cols,2),np.uint8)
mask[crow-isize:crow+isize, ccol-isize:ccol+isize] = 0
# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back2 = cv2.idft(f_ishift)
img_back2 = cv2.magnitude(img_back2[:,:,0],img_back2[:,:,1])


fig, ((axLU, axRU), (axLL, axRL)) = plt.subplots(nrows=2, ncols=2, figsize=(10,6))

axLU.imshow(img, cmap = 'gray')
axLU.set_title('Input image'),
axLU.set_xticks([]), axLU.set_yticks([]) # Remove ticks

axRU.imshow(magnitude_spectrum, cmap = 'gray')
axRU.set_title('Magnitude Spectrum')
axRU.set_xticks([])
axRU.set_yticks([])

axLL.imshow(img_back1, cmap = 'gray')
axLL.set_title('Inverse FFT with Lowpass '),
axLL.set_xticks([])
axLL.set_yticks([]) # Remove ticks

axRL.imshow(img_back2, cmap = 'gray')
axRL.set_title('Inverse FFT with Highpass')
axRL.set_xticks([])
axRL.set_yticks([])

plt.show()
