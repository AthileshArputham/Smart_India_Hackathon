import  tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Activation
from keras.models import model_from_json

model1 = Sequential()

model1.add(Dense(units=32,input_dim=15,activation='tanh'))
model1.add(Dense(units=32,activation='tanh'))
model1.add(Dense(units=21,activation='tanh'))
model1.compile(optimizer='adam',loss='mean_squared_error')

print(model1.predict(x=np.random.random((1,15))))
model1.fit(x=np.random.random((100,15)), y=np.random.random((100,21)),epochs=50)


print(model1.summary())