#
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# resize; https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20resize(InputArray%20src,%20OutputArray%20dst,%20Size%20dsize,%20double%20fx,%20double%20fy,%20int%20interpolation)
# スライダ(trackbar)の位置が変更されたときのコールバック関数
def update(threshVal):
    retVal, img_bin = cv2.threshold(img_gry, threshVal, MAX_VAL, type=cv2.THRESH_BINARY)
    w = int(img_gry.shape[1]*VIEW_SCALE)
    h = int(img_gry.shape[0]*VIEW_SCALE)
    cv2.imshow(WIN_Titile, cv2.resize(img_bin,(w,h)))

if __name__ == '__main__':
    WIN_Titile = 'Binary image'
    VIEW_SCALE = 1.0         # ウィンドウの大きさのスケールファクタ
    DEFAULT_THRESH_VAL = 128 # 閾値処理のデフォルト値
    MAX_VAL = 255            # 8bit 階調を表す

    img_org = cv2.imread('data/lena_std.tif') # オリジナル画像の入力
    img_gry = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY) # グレイスケールに変換

    w = int(img_gry.shape[1]*VIEW_SCALE)
    h = int(img_gry.shape[0]*VIEW_SCALE)

    cv2.imshow('Input image', cv2.resize(img_org,(w, h)))
    cv2.imshow('Grayscale image', cv2.resize(img_gry,(w, h)))

    update(DEFAULT_THRESH_VAL) # 2値値後にウィンドウ表示
    cv2.createTrackbar('threshold', WIN_Titile, DEFAULT_THRESH_VAL, MAX_VAL, update)
    print( "To terminate, hit any key")
    cv2.waitKey(0) #  Program terminate when hit any key.
    cv2.destroyAllWindows()