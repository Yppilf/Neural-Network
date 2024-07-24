import numpy as np
from keras.datasets import mnist # type: ignore
from keras.utils import to_categorical # type: ignore

from losses import mse, mse_prime
from network import Network

from dense import Dense
from activations import Tanh

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, -1)

# # neural network
# networkStructure = [
#     Dense(28 * 28, 40),
#     Tanh(),
#     Dense(40, 10),
#     Tanh()
# ]
# network = Network(networkStructure, learning_rate=0.1)

# # train
# errors = network.train(mse, mse_prime, x_train, y_train, epochs=10, verbose = True)

# # Test some functionalities
# network.dispPrediction(x_test[0])
# network.dispErrors()
# network.saveNetwork("mnist2")

network2 = Network([], overrideInit=True)
network2.loadNetwork("mnist")

# test
correct = 0
for x, y in zip(x_test, y_test):
    output = network2.predict(x)
    prediction = np.argmax(output)
    true_val = np.argmax(y)
    if prediction == true_val:
        correct += 1
testing_accuracy = correct/len(x_test)
print(f"Testing accuracy: {testing_accuracy:.2f}")

