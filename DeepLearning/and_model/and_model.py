# -*- coding: utf-8 -*-
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions

#NNの設定
class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 3)
            self.l2 = L.Linear(None, 2)
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y        

# データの設定
trainx = np.array(([0,0], [0,1], [1,0], [1,1]), dtype=np.float32)
trainy = np.array([0, 0, 0, 1], dtype=np.int32)
train = chainer.datasets.TupleDataset(trainx, trainy)
test = chainer.datasets.TupleDataset(trainx, trainy)

#Chainerの設定
# ニューラルネットワークの登録
model = L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)

chainer.serializers.load_npz("result/out.model", model)

# 学習結果の評価
for i in range(len(trainx)):
    x = chainer.Variable(trainx[i].reshape(1,2))
    result = F.softmax(model.predictor(x))
    print("input: {}, result: {}".format(trainx[i], result.data.argmax()))
