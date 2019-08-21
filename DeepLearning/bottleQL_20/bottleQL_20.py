# -*- coding: utf-8 -*-
import numpy as np

BOTTLE_N = 20
QV0=np.zeros((BOTTLE_N+1,3), dtype=np.float32)
QV1=np.zeros((BOTTLE_N+1,3), dtype=np.float32)
QVs = [QV0, QV1]
pos = 0
pos_old = 0

def step(act, pos, turn):
    pos = pos + act + 1
    rewards = [0,0]
    done = False
    if (pos>=BOTTLE_N):
        pos = BOTTLE_N
        rewards[turn] = -100
        rewards[(turn+1)%2] = 1
        done = True
    return pos, rewards, done

def random_action():
    act = np.random.choice([0, 1, 2])
    return act

epsilon = 0.01

def get_action(pos, episode, QV):
    epsilon = 0.5 * (1 / (episode + 1))
    if np.random.rand()<epsilon:
        return random_action()
    else:
        a = np.where(QV[pos]==QV[pos].max())[0]
        return np.random.choice(a)

def UpdateQTable(act, reward, pos, pos_old, QV):
    alpha = 0.5
    gamma = 0.9
    maxQ = np.max(QV[pos])
    QV[pos_old][act] = (1-alpha)*QV[pos_old][act]+alpha*(reward + gamma*maxQ);

n_episodes = 1000

for i in range(1, n_episodes + 1):
    pos = 0
    pos_old = [0,0]
    rewards = [0,0]
    actions = [0,0]
    while(1):
        actions[0] = get_action(pos, i, QVs[0])
        pos_old[0] = pos
        pos, rewards, done = step(pos, actions[0], 0)
        UpdateQTable(actions[1], rewards[1], pos, pos_old[1], QVs[1])
        if (done==True):
            UpdateQTable(actions[0], rewards[0], pos, pos_old[0], QVs[0])
            print('{} : 0 Loose, 1 Win!!'.format(i))
            break
        actions[1] = get_action(pos, i, QVs[1])
        pos_old[1] = pos
        pos, rewards, done = step(pos, actions[1], 1)
        UpdateQTable(actions[0], rewards[0], pos, pos_old[0], QVs[0])
        if (done==True):
            UpdateQTable(actions[1], rewards[1], pos, pos_old[1], QVs[1])
            print('{} : 0 Win!!, 1 Loose'.format(i))
            break
print("Agent 0")
print(QVs[0])
print("Agent 1")
print(QVs[1])
np.savetxt('QV0.txt', QVs[0])
np.savetxt('QV1.txt', QVs[1])
