import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense

trX = np.linspace(-1,1,101)
trY = 3*trX + np.random.randn(*trX.shape)*0.33

model = Sequential()

model.add(Dense(input_dim=1, units=1, kernel_initializer='uniform', activation='linear'))

#model.fit(trX, trY, batch_size=32, epochs=10, validation_data=(x_val, y_val))

weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Linear regression model is trained to have weights w: %.2f, b: %.2f' % (w_init, b_init))

model.compile(optimizer='sgd', loss='mse')
model.fit(trX, trY, epochs=200, verbose=1)

weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b:%.2f' % (w_final, b_final))

model.save_weights('my_model.h5')
#model.load_weights('my_model.h5')
