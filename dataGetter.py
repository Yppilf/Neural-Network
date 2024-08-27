import numpy as np
from keras.datasets import mnist # type: ignore
from keras.utils import to_categorical # type: ignore
import os, h5py

class DataGetter():
    def __init__(self):
        x_train = []
        y_train = []
        x_test = []
        y_test = []

    def load_mnist(self, trainLim=-1, testLim=-1):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train, self.y_train = self.preprocess_data(x_train, y_train, trainLim)
        self.x_test, self.y_test = self.preprocess_data(x_test, y_test, testLim)

    def preprocess_data(self, x, y, limit):
        # reshape and normalize input data
        x = x.reshape(x.shape[0], 28 * 28, 1)
        x = x.astype("float32") / 255
        # encode output which is a number in range [0,9] into a vector of size 10
        # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        y = to_categorical(y)
        y = y.reshape(y.shape[0], 10, 1)
        return x[:limit], y[:limit]
    
    def write_files(self, folder_name):
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, 'data.h5')
        
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('x_train', data=self.x_train)
            f.create_dataset('y_train', data=self.y_train)
            f.create_dataset('x_test', data=self.x_test)
            f.create_dataset('y_test', data=self.y_test)

    def load_files(self, folder_name):
        file_path = os.path.join(folder_name, 'data.h5')
        
        with h5py.File(file_path, 'r') as f:
            self.x_train = f['x_train'][:]
            self.y_train = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test = f['y_test'][:]

if __name__ == "__main__":
    dg = DataGetter()
    # dg.load_mnist(1000,100)
    # dg.write_files("mnist_data")

    dg.load_mnist(10,3)
    dg.write_files("mini_mnist_data")
