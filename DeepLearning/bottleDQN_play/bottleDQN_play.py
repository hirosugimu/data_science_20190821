# -*- coding: utf-8 -*-
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import copy


BOTTLE_N = 20
BT=np.zeros(BOTTLE_N, dtype=np.float32)
TEST=np.zeros(BOTTLE_N, dtype=np.float32)
pos = 0
pos_old = 0

def step(act, pos, turn):
    pos = pos + act + 1
    rewards = [0,0]
    done = False
    if (pos>=BOTTLE_N):
        pos = BOTTLE_N
        rewards[turn] = -1
        rewards[(turn+1)%2] = 1
        done = True
    return pos, rewards, done

def random_action():
    act = np.random.choice([0, 1, 2])
    return act

def champion_action(pos):
    if np.random.rand()<0.1:
        act = np.random.choice([0, 1, 2])
    else:
        if (BOTTLE_N-pos)%4==1:
            act = np.random.choice([0, 1, 2])
        elif (BOTTLE_N-pos)%4==0:
            act = 2
        elif (BOTTLE_N-pos)%4==3:
            act = 1
        elif (BOTTLE_N-pos)%4==2:
            act = 0
    return act

class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=100):
        w = chainer.initializers.HeNormal(scale=1.0) 
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l0=L.Linear(obs_size, n_hidden_channels, initialW=w)
            self.l1=L.Linear(n_hidden_channels, n_hidden_channels, initialW=w)
            self.l2=L.Linear(n_hidden_channels, n_actions, initialW=w)

    def __call__(self, x, test=False):        
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))

q_func = QFunction(BOTTLE_N, 3)

optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
q_func.to_cpu()

explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.01, decay_steps=1000, random_action_func=random_action)

#replay_buffer0 = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=10 ** 6)
replay_buffer1 = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=10 ** 6)

gamma = 0.95

#agent0 = chainerrl.agents.DoubleDQN(
#    q_func, optimizer, replay_buffer0, gamma, explorer, minibatch_size = 100,
#    replay_start_size=100, update_interval=1, target_update_interval=100)
agent1 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer1, gamma, explorer, minibatch_size = 100,
    replay_start_size=100, update_interval=1, target_update_interval=100)

agent1.load('agent1')

n_episodes = 1

for i in range(1, n_episodes + 1):
    BT[0:BOTTLE_N] = 0
    pos = 0
    pos_old = 0
    rewards = [0,0]
    actions = [0,0]
    bt = BT.copy()
    print('New Game pin:{}'.format(BOTTLE_N - pos))
    while(1):
        actions[0] = int(input('[1-3]'))
        pos_old = pos
        pos, rewards, done = step(actions[0], pos, 0)
        print('act:{0}, pin:{1}'.format(actions[0], BOTTLE_N - pos))
        BT[pos_old:pos] = 1
        bt = BT.copy()
        if (done==True):
            print('You loose')
            break
        actions[1] = agent1.act(bt)
        pos_old = pos
        pos, rewards, done = step(actions[1], pos, 1)
        print('act:{0}, pin:{1}'.format(actions[1], BOTTLE_N - pos))
        BT[pos_old:pos] = 1
        bt = BT.copy()
        if (done==True):
            print('You win!!')
            break
