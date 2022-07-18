# UnknownError: Failed to get convolution algorithm. 
#This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
# 해당 에러가 뜰때 해결법
# 처음에는 메모리를 조금만 할당하고, 프로그램이 실행되어 더 많은 GPU 메모리가 필요하면, 
# 텐서플로 프로세스에 할당된 GPU 메모리 영역을 확장할 수 있게 해줘야 한다

# [ERROR] 해결법   (https://modernflow.tistory.com/9)
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    


# START    
# CNN MODEL    
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist

# 데이터 불러오기 및 핸들링
(x_train,y_train),(x_test,y_test)=mnist.load_data()
num_labels=len(np.unique(y_train))
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
shape1=x_train.shape[1]
shape2=x_train.shape[2]
x_train=x_train.reshape(-1, shape1,shape2,1).astype('float32')
x_test=x_test.reshape(-1, shape1,shape2,1).astype('float32')
x_train=x_train/255.
x_test=x_test/255.
#표본수를 포함하여 4D 텐서이므로 입력은 3D 텐서가 된다.
input_shape=(shape1,shape2,1)
batch_size=64
kernel_size=3
pool_size=2
filters=64
dropout=0.3

#CNN 모델 설계
model=Sequential()
model.add(Conv2D(filters=filters,kernel_size=kernel_size,activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,kernel_size=kernel_size,activation='relu'))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,kernel_size=kernel_size,activation='relu'))
model.add(Flatten())
model.add(Dropout(dropout))
model.add(Dense(num_labels, activation='softmax'))
model.summary()
plot_model(model,to_file='E:/박사 과정/대학원 수업자료/2021_2R/시뮬레이션 컴퓨팅 2/딥러닝프로그램/py 형식/제4장/ch4_2.png', show_shapes=True)



# validation set
x_val=x_train[:10000]
partial_x_train=x_train[10000:]
y_val=y_train[:10000]
partial_y_train=y_train[10000:]
print(partial_x_train.shape)
print(partial_y_train.shape)

# validationset을 이용해 적합
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
history=model.fit(partial_x_train, partial_y_train, epochs=100, batch_size=64,validation_data=(x_val,y_val))
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
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=4,batch_size=64)

result_train=model.evaluate(x_train,y_train)
result_test=model.evaluate(x_test,y_test)
print(result_train)
print(result_test)
pred=model.predict(x_test)
pred=pd.DataFrame(pred)
pred.head()




