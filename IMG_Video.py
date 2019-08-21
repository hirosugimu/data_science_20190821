# -*- coding: utf-8 -*-
# Capture Video from Camera
# ref.: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
#
import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(0)
print( "To terminate, hit Escape key")
if not cap.isOpened():
    print("Fail to open videocapture")
    sys.exit()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # グレースケール表示にしたい場合
    # Our operations on the frame come here
    # もし、カメラが無いと、次の実行で 次のようなエラーが生じる。
    # "error: (-215) scn == 3 || scn == 4 in function cv::ipp_cvtColor" 
    # このエラーの参照：http://answers.opencv.org/question/90613/error-215-scn-3-scn-4-in-function-cvipp_cvtcolor/
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    cv2.imshow('frame',gray)

    cv2.imshow('frame',frame) # Display color frame
    
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
# ESCキー押下で終了
    if cv2.waitKey(30) & 0xff == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()