# -*- coding: utf-8 -*-
# Wiki https://en.wikipedia.org/wiki/Haar-like_feature 
# OpenCV-Python　https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html　

import cv2
import sys

# Haar-like特徴分類器の読み込み
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

# 顔データの読込み
img = cv2.imread('data/face.png')

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔を検知
faces = face_cascade.detectMultiScale(gray)

for (x,y,w,h) in faces:
    # 検知した顔を矩形で囲む
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # 顔画像（グレースケール）
    roi_gray = gray[y:y+h, x:x+w]
    # 顔画像（カラースケール）
    roi_color = img[y:y+h, x:x+w]
    # 顔の中から目を検知
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        # 検知した目を矩形で囲む
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# 画像表示
cv2.imshow('img',img)

# 何かキーを押したら終了
cv2.waitKey(0)
cv2.destroyAllWindows()