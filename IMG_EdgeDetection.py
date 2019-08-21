# -*- coding: utf-8 -*-
# Edge Detection based on Canny method
# Usage Canny ,  two threshold are defined based on 
# http://opencv.jp/opencv-2.1/cpp/feature_detection.html?highlight=canny#Canny
# http://en.wikipedia.org/wiki/Canny_edge_detector
# http://opencv.jp/sample/gradient_edge_corner.html
# http://opencv.jp/opencv-2svn/cpp/imgproc_image_filtering.html

# CV_64F: 出力画像のbit深度
# https://docs.opencv.org/2.4.13.4/modules/core/doc/intro.html
# https://docs.opencv.org/3.1.0/dc/dcc/cvdef_8h.html

import cv2

def EdgeDetections(f_name):
#    img = cv2.imread( f_name, 0 ) # 0 means read image as GRAY_SCALE
    img_org = cv2.imread(f_name)
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY) # グレイスケールに変換
    imgsize = img.size
    
    edge_sob_x = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=5)
    edge_lapl = cv2.Laplacian(img, cv2.CV_32F) 
    edge_cann = cv2.Canny(img, 80, 120)
    cv2.imshow('Original Image', img_org)
    cv2.imshow('Gray scale', img)
    cv2.imshow('Sobel', edge_sob_x)
    cv2.imshow('Laplacian', edge_lapl)
    cv2.imshow('Canny', edge_cann)
    print( "To terminate, hit any key")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
#    f_name = 'data/IP_Magic_Square.png'
    f_name = 'data/test.jpg'
    rc = EdgeDetections(f_name)