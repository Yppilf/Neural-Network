# Neural Network From Scratch

The base code and inspiration is from YouTube: [Neural Network from Scratch | Mathematics & Python Code](https://youtube.com/playlist?list=PLQ4osgQ7WN6PGnvt6tzLAVAEMsL3LBqpm).

# Edits
Added docstrings
Fixed MNIST imports
Converted to object oriented approach
Added save and load functionality
CNN code does not work currently, only the neural network does. See mnist.py for example codes. Use mnist.json for a model on the MNIST database

# C++  version
In the folder "CPPversion", a lot of the functionality of the python module has been written in C++ for faster execution. 
The code does not require any external modules outside of the default and should function by simply running "make" in the directory on a linux machine (otherwise untested)
Strangely in the current version the code works fine for the mini_mnist_data, but immediately throws a segmentation fault for larger folders, without changing any other code. I do not know why this happens.