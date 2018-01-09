# -*- coding: utf-8 -*-
# @Time    : 2018/1/9 16:26
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : train.py
# @Software: PyCharm

from data.dataload import dataload
from models.ResNet50 import ResNet50
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
model = ResNet50(input_shape=(32, 32, 3), classes=10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#load data:cifar10
X_train, Y_train, X_test, Y_test = dataload()


model.fit(X_train, Y_train, epochs=20, batch_size=32)

preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
