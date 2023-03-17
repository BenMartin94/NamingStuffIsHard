import keras
from keras.layers import Dense, Dropout, Activation
from Utils import *
from keras.callbacks import EarlyStopping
import pickle

pickleFile = "./pairs.pickle"
with open(pickleFile, 'rb') as handle:
    pairs = pickle.load(handle)
    uniqueMoves = pickle.load(handle)

# create numpy vectors for training
X = np.zeros((len(pairs), 8, 8))
y = np.zeros((len(pairs), len(uniqueMoves)))
for i in range(len(pairs)):
    X[i] = pairs[i].position
    y[i][pairs[i].moveNumber] = 1

# create convolutional model
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(8, 8, 1)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# now flatten and do a dense layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(len(uniqueMoves)))
model.add(keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.summary()

model.fit(X, y, epochs=200, batch_size=128, validation_split=0.2, callbacks=[EarlyStopping(patience=20)])
model.save("NaiveClassifier")
