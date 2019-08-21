# -*- coding: utf-8 -*-
# OpenCV-Python https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_lucas_kanade.html
# OpenCV　https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html　
# この日本語：　http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
# Shi-Tomasi法、Lucas-Kanade法
import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(0)
print( "To terminate, hit Escape key")
if not cap.isOpened():
    print("Fail to open videocapture")
    sys.exit()

# ShiTomasi corner detection パラメータの設定
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# lucas kanade optical flow パラメータの設定
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# ランダムに100色生成（値0～255の階調，100行3列のndarrayを生成）
color = np.random.randint(0, 255, (100, 3))

# 最初のフレーム処理
end_flag, frame = cap.read()
gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
feature_prev = cv2.goodFeaturesToTrack(gray_prev, mask = None, **feature_params)
mask = np.zeros_like(frame)

while(end_flag):
    # グレースケールに変換
    gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # オプティカルフロー検出
    feature_next, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_next, feature_prev, None, **lk_params)

    # オプティカルフローを検出した特徴点を選別（0：検出せず、1：検出した）
    good_prev = feature_prev[status == 1]
    good_next = feature_next[status == 1]

    # オプティカルフローを描画
    for i, (next_point, prev_point) in enumerate(zip(good_next, good_prev)):
        prev_x, prev_y = prev_point.ravel()
        next_x, next_y = next_point.ravel()
        mask = cv2.line(mask, (next_x, next_y), (prev_x, prev_y), color[i].tolist(), 2)
        frame = cv2.circle(frame, (next_x, next_y), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    # ウィンドウ表示
    cv2.imshow('window', img)

    # ESCキー押下で終了
    if cv2.waitKey(30) & 0xff == 27:
        break

    #次のフレーム、ポイントの準備
    gray_prev = gray_next.copy()
    feature_prev = good_next.reshape(-1, 1, 2)
    end_flag, frame = cap.read()

# 終了処理
cv2.destroyAllWindows()
cap.release()