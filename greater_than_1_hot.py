import numpy as np
from keras.models import Sequential
from keras.layers import Dense

seed = 7 # Random seed chosen at random, guaranteed 100% random results
np.random.seed(seed)
DATA_SIZE = 14000

train_input = np.random.rand(DATA_SIZE, 2)
train_output = np.asarray(map(lambda x: [1, 0] if x[0] > x[1] else [0,1] , train_input))
test_input = np.random.rand(DATA_SIZE, 2) * 100
test_output = map(lambda x: [1, 0] if x[0] > x[1] else [0, 1], test_input)

model = Sequential()
# More layers make it harder for the network to learn
model.add(Dense(5, input_dim=2, kernel_initializer='normal', activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation='relu'))
model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_input, train_output, epochs=20, batch_size=100, verbose=2)
scores = model.evaluate(test_input, test_output, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

test_model_output = model.predict(test_input)

# for i in range(len(test_input)):
#     if np.argmax(test_model_output[i]) != np.argmax(test_input[i]):
#         print str(np.argmax(test_model_output[i])) + " " + str(np.argmax(test_input[i]))
#         print str(test_model_output[i]) + " " + str(test_input[i])