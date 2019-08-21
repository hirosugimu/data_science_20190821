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
iris = load_iris()
data_train, data_test, label_train, label_test = train_test_split(iris.data, iris.target, test_size=0.2)
data_train = (data_train).astype(np.float32)
data_test = (data_test).astype(np.float32)
train = chainer.datasets.TupleDataset(data_train, label_train)
test = chainer.datasets.TupleDataset(data_test, label_test)

#Chainerの設定
# ニューラルネットワークの登録
model = L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
# イテレータの定義
train_iter = chainer.iterators.SerialIterator(train, batchsize)# 学習用
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)# 評価用
# アップデータの登録
updater = training.StandardUpdater(train_iter, optimizer)
# トレーナーの登録
trainer = training.Trainer(updater, (epoch, 'epoch'))

# 学習状況の表示や保存
trainer.extend(extensions.LogReport())#ログ
trainer.extend(extensions.Evaluator(test_iter, model))# エポック数の表示
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time'] ))#計算状態の表示
#trainer.extend(extensions.dump_graph('main/loss'))#ニューラルネットワークの構造
#trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch',file_name='loss.png'))#誤差のグラフ
#trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'epoch', file_name='accuracy.png'))#精度のグラフ
#trainer.extend(extensions.snapshot(), trigger=(100, 'epoch'))# 再開のためのファイル出力

#chainer.serializers.load_npz("result/snapshot_iter_500", trainer)#再開用

# 学習開始
trainer.run()

# 途中状態の保存
chainer.serializers.save_npz("result/Iris.model", model)
