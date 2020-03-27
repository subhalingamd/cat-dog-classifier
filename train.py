import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout

# Dataset
x=np.load('dataset-photos.npy')
y=np.load('dataset-labels.npy')

model=Sequential()
model.add(Conv2D(filters=16,kernel_size=(5,5),activation='relu',padding='same',input_shape=x.shape[1:]))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=48,kernel_size=(2,2),activation='relu',padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=5, batch_size=32)

model.save('model.h5')
