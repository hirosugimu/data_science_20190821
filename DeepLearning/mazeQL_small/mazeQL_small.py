# -*- coding: utf-8 -*-
import numpy as np
import cv2

MAP_X = 5
MAP_Y = 4

MAP = np.array([
-1, -1, -1, -1, -1,
-1,  0,  0,  0, -1,
-1,  0, -1,  1, -1,
-1, -1, -1, -1, -1,
], dtype=np.float32)

QV=np.zeros((MAP_X* MAP_Y,4), dtype=np.float32)

img = np.zeros((480,640,1), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

pos = 7
pos_old = pos

def reset():
    global pos, pos_old
    pos = 7
    pos_old = pos

def step(act):
    global pos, pos_old
    pos_old = pos
    x = pos%MAP_X
    y = pos//MAP_X
    if (act==0):
        y = y-1
    elif (act==1):
        x = x + 1
    elif (act==2):
        y = y + 1
    elif (act==3):
        x = x - 1

    if (x<0 or y<0 or x>=MAP_X or y>=MAP_Y):
        pos = pos_old
        reward = -1
    else:
        pos = x+y*MAP_X
        reward = MAP[pos]
    return reward

def random_action():
    act = np.random.choice([0, 1, 2, 3])
    return act

def get_action():
    global pos
    epsilon = 0.01
    if np.random.rand()<epsilon:
        return random_action()
    else:
        a = np.where(QV[pos]==QV[pos].max())[0]
        return np.random.choice(a)

def UpdateQTable(act, reward):
    global pos, pos_old, QV
    alpha = 0.2
    gamma = 0.9
    maxQ = np.max(QV[pos])
    QV[pos_old][act] = (1-alpha)*QV[pos_old][act]+alpha*(reward + gamma*maxQ);

def disp():
    global pos
    img.fill(255)
    d = 480//MAP_X
    for s in range(0, MAP_X*MAP_Y):
        x = (s%MAP_X)*d
        y = (s//MAP_X)*d
        if MAP[s]==-1:
            cv2.rectangle(img,(x,y),(x+d,y+d),0,-1)
        cv2.rectangle(img,(x,y),(x+d,y+d),0,1)
    x = (pos%MAP_X)*d
    y = (pos//MAP_X)*d
    cv2.circle(img,(x+d//2,y+d//2),int(d//2*0.8),32,5)
    if (MAP[pos]==-1):
        cv2.circle(img,(x+d//2,y+d//2),int(d//2*0.8),224,5)
    else:
        cv2.circle(img,(x+d//2,y+d//2),int(d//2*0.8),32,5)
    for s in range(0, MAP_X*MAP_Y):
        x = (s%MAP_X)*d
        y = (s//MAP_X)*d
        for a in range(0, 4):
            cv2.putText(img,str('%03.3f' % QV[s][a]),(x+1,y+(a+1)*(d//5)), font, 0.3,127,1)
    cv2.imshow('res',img)
    cv2.waitKey(1)

n_episodes = 1000
n_steps = 1000
for i in range(1, n_episodes + 1):
    reset()
    for i in range(1, n_steps + 1):
        disp()
        action = get_action()
        reward = step(action)
        UpdateQTable(action, reward)
        if (reward==1):
            break

