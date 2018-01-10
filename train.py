# -*- coding: utf-8 -*-
# @Time    : 2018/1/9 16:26
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : train.py
# @Software: PyCharm

from data.dataload import dataload
from models.ResNet50 import ResNet50
# from models.ResNet50_cl import ResNet50
import os
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)

os.environ['CUDA_VISIBLE_DEVICES']='0'
model = ResNet50(input_shape=(32, 32, 3), classes=10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#load data:cifar10
X_train, Y_train, X_test, Y_test = dataload()


model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=32,show_accuracy = True,
          shuffle=True,
          callbacks=[lr_reducer, early_stopper])

preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
