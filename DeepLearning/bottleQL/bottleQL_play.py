# -*- coding: utf-8 -*-
import numpy as np

BOTTLE_N = 9
QV=np.zeros((BOTTLE_N+1,3), dtype=np.float32)
pos = 0
pos_old = 0

def step(act, pos, turn):
    pos = pos + act
    rewards = [0,0]
    done = False
    if (pos>=BOTTLE_N):
        pos = BOTTLE_N
        rewards[turn] = -100
        rewards[(turn+1)%2] = 1
        done = True
    return pos, rewards, done

def random_action():
    act = np.random.choice([1, 2, 3])
    return act

epsilon = 0.01

def get_action(pos, episode):
    epsilon = 0#0.5 * (1 / (episode + 1))
    if np.random.rand()<epsilon:
        return random_action()
    else:
        a = np.where(QV[pos]==QV[pos].max())[0]
        return np.random.choice(a)+1

def UpdateQTable(act, reward, pos, pos_old):
    alpha = 0.5
    gamma = 0.9
    maxQ = np.max(QV[pos])
    QV[pos_old][act] = (1-alpha)*QV[pos_old][act]+alpha*(reward + gamma*maxQ);

n_episodes = 1

QV = np.loadtxt('QV1.txt')

for i in range(1, n_episodes + 1):
    pos = 0
    pos_old = 0
    rewards = [0,0]
    pos2 = [0,0]
    print('New Game pin:{}'.format(BOTTLE_N - pos))
    while(1):
        action = int(input('[1-3]'))
        pos, rewards, done = step(pos, action, 1)
        print('act:{0}, pin:{1}'.format(action, BOTTLE_N - pos))
        if (done==True):
            print('You Loose!')
            break
        action = get_action(pos, i)
        pos_old = pos
        pos, rewards, done = step(pos, action, 0)
        print('act:{0}, pin:{1}'.format(action, BOTTLE_N - pos))
        if (done==True):
            print('You Win!')
            break
