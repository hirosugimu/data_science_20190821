# -*- coding: utf-8 -*-
#
# A-KAZE algorithm: copyright described in https://github.com/pablofdezalc/akaze/blob/master/LICENSE
#
import cv2
import numpy as np

# image 1
img1 = cv2.imread("data/book.jpg")
#cv2.imshow('img1', img1)
# image2
img2 = cv2.imread("data/book_in_scene.jpg")

# A-KAZE検出器の生成
# https://docs.opencv.org/3.4.1/d8/d30/classcv_1_1AKAZE.html
akaze = cv2.AKAZE_create()                                

# 各画像から特徴量の検出と特徴量ベクトルの計算
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)

# Brute-Force Matcher生成
bf = cv2.BFMatcher()

# 特徴量ベクトル同士をBrute-Force+kNNでマッチング
matches = bf.knnMatch(des1, des2, k=2)

# データを間引きする, ratioと，特徴点同士を結ぶ対応線の数は比例する
ratio = 0.6
good = []
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append([m])

# 対応する特徴点を直線で引く（対応線）
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# 画像表示
cv2.imshow('img', img3)

# キー押下で終了
print( "To terminate, hit any key")
cv2.waitKey(0)
cv2.destroyAllWindows()