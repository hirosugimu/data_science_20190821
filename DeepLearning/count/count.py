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
            self.l2 = L.Linear(None, 4)
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y        

# データの設定
trainx = np.array(([0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]), dtype=np.float32)
trainy = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.int32)
train = chainer.datasets.TupleDataset(trainx, trainy)
test = chainer.datasets.TupleDataset(trainx, trainy)

#Chainerの設定
# ニューラルネットワークの登録
model = L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
# イテレータの定義
batchsize = 4
train_iter = chainer.iterators.SerialIterator(train, batchsize)# 学習用
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)# 評価用
# アップデータの登録
updater = training.StandardUpdater(train_iter, optimizer)
# トレーナーの登録
epoch=2000
trainer = training.Trainer(updater, (epoch, 'epoch'))

# 学習状況の表示や保存
trainer.extend(extensions.LogReport())#ログ
trainer.extend(extensions.Evaluator(test_iter, model))# エポック数の表示
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time'] ))#計算状態の表示
#trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch',file_name='loss.png'))#誤差のグラフ
#trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'epoch', file_name='accuracy.png'))#精度のグラフ

# 学習開始
trainer.run()

#モデルの保存
chainer.serializers.save_npz("result/out.model", model)
