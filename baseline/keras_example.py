from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import numpy as np

np.random.seed(100)

model = Sequential()
model.add(Dense(input_dim=2, output_dim=16))
model.add(Activation('sigmoid'))
model.add(Dense(input_dim=16, output_dim=2))
model.add(Activation('softmax'))
X = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
y = np.array([[0,1],[0,1],[0,1],[1,0]], "float32")
sgd = SGD(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=sgd)
model.fit(X, y, nb_epoch=3000, batch_size=1)
test_X=np.array([[0,0],[0,1],[1,0],[1,1]],"float32")
print(model.predict_proba(test_X))
print(model.predict_classes(test_X))
print(model.predict(test_X))
