# -*- coding: utf-8 -*-
import numpy as np
import cv2
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import copy

MAP_X = 10
MAP_Y = 10

#ñ¿òHÇÃê›íË
MAP = np.array([
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
-1,  0,  0,  0, -1,  0,  0,  0,  0, -1,
-1,  0, -1,  0,  0,  0, -1, -1,  0, -1,
-1,  0, -1, -1,  0, -1, -1,  0,  0, -1,
-1, -1, -1,  0,  0,  0, -1, -1,  0, -1,
-1,  0, -1,  0, -1,  0, -1,  0,  0, -1,
-1,  0,  0,  0, -1,  0,  0,  0, -1, -1,
-1,  0, -1,  0, -1, -1,  0, -1, -1, -1,
-1,  0,  0,  0, -1,  0,  0,  0,  1, -1,
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
], dtype=np.float32)

img = np.zeros((480,640,1), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

pos = 12
pos_old = pos

def reset():
    global pos, pos_old
    pos = 12
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
    map = copy.copy(MAP)
    for s in range(0, MAP_X*MAP_Y):
        x = (s%MAP_X)*d
        y = (s//MAP_X)*d
        tmp = map[s]
        map[s] = 10
        act = agent.act(map)
        map[s] = tmp
        cv2.putText(img,str('%d' % act),(x+1,y+d), font, 1,127,1)
    cv2.imshow('res',img)
    cv2.waitKey(1)

# Q-ä÷êîÇÃíËã`
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=256):
        super(QFunction, self).__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_hidden_channels),
            l3=L.Linear(n_hidden_channels, n_actions))

    def __call__(self, x, test=False):        
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = F.leaky_relu(self.l2(h))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))

q_func = QFunction(MAP_X*MAP_Y, 4)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
q_func.to_cpu()

explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.1, decay_steps=100, random_action_func=random_action)

replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=10 ** 6)

gamma = 0.95

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=50, update_interval=1, target_update_interval=10)

n_episodes = 100
n_steps = 1000
for i in range(1, n_episodes + 1):
    reset()
    reward = 0
    done = False
    for j in range(1, n_steps + 1):
        disp()
        map = copy.copy(MAP)
        map[pos] = 10
        action = agent.act_and_train(map, reward)
        reward = step(action)
        if (reward==1):
            done = True
            break
    print(i, j)
    map = copy.copy(MAP)
    map[pos] = 10
    agent.stop_episode_and_train(map, reward, done)
    
agent.save('agent')
