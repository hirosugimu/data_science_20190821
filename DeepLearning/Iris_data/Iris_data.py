# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
data_train, data_test, label_train, label_test = train_test_split(iris.data, iris.target, test_size=0.1)
np.savetxt('iris_train_data.txt', data_train,delimiter=',')
np.savetxt('iris_train_label.txt', label_train,delimiter=',')
np.savetxt('iris_test_data.txt', data_test,delimiter=',')
np.savetxt('iris_test_label.txt', label_test,delimiter=',')
