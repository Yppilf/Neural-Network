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
    
    def write_3d_array_to_csv(self, filename, array):
        with open(filename, 'w') as file:
            for slice_ in array:
                for row in slice_:
                    row_str = ','.join(map(str, row))
                    file.write(f"{row_str}\n")
                file.write("\n")  # Separate slices by a blank line

    def write_files(self, folder_name):
        os.makedirs(folder_name, exist_ok=True)
        self.write_3d_array_to_csv(f"{folder_name}/x_train.csv", self.x_train)
        self.write_3d_array_to_csv(f"{folder_name}/y_train.csv", self.y_train)
        self.write_3d_array_to_csv(f"{folder_name}/x_test.csv", self.x_test)
        self.write_3d_array_to_csv(f"{folder_name}/y_test.csv", self.y_test)

    def read_3d_array_from_csv(self, filename):
        with open(filename, 'r') as file:
            slices = []
            slice_ = []

            for line in file:
                line = line.strip()
                if not line:
                    if slice_:
                        slices.append(slice_)
                        slice_ = []
                else:
                    row = list(map(float, line.split(',')))
                    slice_.append(row)

            if slice_:
                slices.append(slice_)

        return np.array(slices)

    def load_files(self, folder_name):
        self.x_train = self.read_3d_array_from_csv(f"{folder_name}/x_train.csv")
        self.y_train = self.read_3d_array_from_csv(f"{folder_name}/y_train.csv")
        self.x_test = self.read_3d_array_from_csv(f"{folder_name}/x_test.csv")
        self.y_test = self.read_3d_array_from_csv(f"{folder_name}/y_test.csv")

if __name__ == "__main__":
    dg = DataGetter()
    dg.load_mnist(1000,100)
    dg.write_files("mnist_data_2")
