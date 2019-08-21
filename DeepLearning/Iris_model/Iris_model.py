# -*- coding: utf-8 -*-
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

#NNの設定
class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(4, 10)
            self.l2 = L.Linear(10, 10)
            self.l3 = L.Linear(10, 3)
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y        

epoch = 1000
batchsize = 100

# データの設定
with open('iris_test_data.txt', 'r') as f:
    lines = f.readlines()
data = []
for l in lines:
    d = l.strip().split(',')
    data.append(list(map(float, d)))
test = np.array(data, dtype=np.float32)

with open('iris_test_label.txt', 'r') as f:
    lines = f.readlines()
data = []
for l in lines:
    d = l.strip().split()
    data.append(list(map(float, d)))
label = np.array(data, dtype=np.int32)[:,0]

#Chainerの設定
# ニューラルネットワークの登録
model = L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)
#学習モデルの読み込み
chainer.serializers.load_npz("result/Iris.model", model)

# 学習結果の評価
for i in range(len(test)):
    x = chainer.Variable(test[i].reshape(1,4))
    result = F.softmax(model.predictor(x))
    print("input: {}, result: {}, ans: {}".format(test[i], result.data.argmax(), label[i]))
