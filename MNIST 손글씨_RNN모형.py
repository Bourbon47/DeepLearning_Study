#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input, SimpleRNN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
shape1=x_train.shape[1]
shape2=x_train.shape[2]
x_train=x_train.reshape(-1, shape1,shape2).astype('float32') #RNN의 입력은 표본을 포함하여 3D 텐서임
x_test=x_test.reshape(-1, shape1,shape2).astype('float32')
x_train=x_train/255.
x_test=x_test/255.
input_shape=(shape1,shape2) #RNN의 표본 하나당 입력은 2D 텐서임
model=Sequential()
model.add(SimpleRNN(units=256,dropout=0.2, input_shape=input_shape))
model.add(Dense(10,activation='softmax'))
model.summary()
plot_model(model,to_file='E:/박사 과정/대학원 수업자료/2021_2R/시뮬레이션 컴퓨팅 2/딥러닝프로그램/py 형식/제4장/ch4_3.png', show_shapes=True)


# validation set
x_val=x_train[:10000]
partial_x_train=x_train[10000:]
y_val=y_train[:10000]
partial_y_train=y_train[10000:]
print(partial_x_train.shape)
print(partial_y_train.shape)

# validationset을 이용해 적합
model.compile(loss='categorical_crossentropy',optimizer='RMSprop', metrics=['accuracy'])
history=model.fit(partial_x_train, partial_y_train, epochs=2, batch_size=64,validation_data=(x_val,y_val))
history_out=history.history
history_out.keys()


# 튜닝한 모형 그림
import matplotlib.pyplot as plt
loss=history_out['loss']
loss_val=history_out['val_loss']
accuracy=history_out['accuracy']
accuracy_val=history_out['val_accuracy']
plt.plot(loss,'bo',label='training loss')
plt.plot( loss_val,'b', label='validation loss' )
plt.title('Training and validation losss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(accuracy,'bo',label='training loss')
plt.plot(accuracy_val,'b', label='validation loss' )
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# 위에서 TRAIN 40000, VAL 10000 했을 때 모델이 잘 나왔으므로, VAL을 제외하고 TRAIN 전체를 이용한 모델 적합
import pandas as pd
model.compile(loss='categorical_crossentropy', optimizer='RMSprop',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,batch_size=64)

result_train=model.evaluate(x_train,y_train)
result_test=model.evaluate(x_test,y_test)
print(result_train)
print(result_test)
pred=model.predict(x_test)
pred=pd.DataFrame(pred)
pred.head()














