# coding:utf-8
import cv2

n1 = 0
n2 = 0
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    xp = int(frame.shape[1]/2)
    yp = int(frame.shape[0]/2)
    d = 40
    cv2.rectangle(gray, (xp-d, yp-d), (xp+d, yp+d), color=0, thickness=2)
    cv2.imshow('gray', gray)
    c =cv2.waitKey(10) 
    if c == 97:
        cv2.imwrite('img/a/{0}.png'.format(n1), gray[yp-d:yp + d, xp-d:xp + d])
        n1 = n1 + 1
    elif c == 98:
        cv2.imwrite('img/b/{0}.png'.format(n2), gray[yp-d:yp + d, xp-d:xp + d])
        n2 = n2 + 1
    elif c == 113:
        break
cap.release()
