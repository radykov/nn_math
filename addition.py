import numpy as np
from keras.models import Sequential
from keras.layers import Dense

seed = 7 # Random seed chosen at random, guaranteed 100% random results
np.random.seed(seed)
DATA_SIZE = 10000

train_input = np.random.rand(DATA_SIZE, 2)
train_output = np.sum(train_input, axis=1)
test_input = np.random.rand(DATA_SIZE, 2) * 100
test_output = np.sum(test_input, axis=1)

model = Sequential()
model.add(Dense(5, input_dim=2, kernel_initializer='normal', activation='relu'))
model.add(Dense(5, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(train_input, train_output, epochs=10, batch_size=100, verbose=2)
scores = model.evaluate(test_input, test_output, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


test_model_output = model.predict(test_input)

for i in range(len(test_input)):
    if np.argmax(test_model_output[i]) != np.argmax(test_input[i]):
        print str(np.argmax(test_model_output[i])) + " " + str(np.argmax(test_input[i]))
        print str(test_model_output[i]) + " " + str(test_input[i])