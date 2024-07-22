def predict(network, input):
    """Forward propagates the entire network
    
    Parameters:
    network (list)  - List containing the layers of classes and activation functions
    input (list)    - List containing the input to the first layer of the network
    
    Returns:
    (list) - output of the network"""
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    """Trains a given network on a given training set with given inputs
    
    Parameters:
    network         (list)  - List containing the layers of classes and activation functions
    loss            (func)  - Function used to compute the loss, used for computing the errors
    loss_prime      (func)  - Function for computing the derivative of the loss function, used for gradient descent
    x_train         (list)  - List containing the input variables for the neural network
    y_train         (list)  - List containing the labels for the training data given in x_train
    epochs          (int)   - The amount of training cycles used. Default = 1000
    learning_rate   (float) - Scale factor for updating training parameters. Default = 0.01
    verbose         (bool)  - Whether to print output or not. Default = True
    
    Returns:
    (list) - Errors of the network at each epoch"""
    errors = []
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        errors.append(error)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
    return errors
        
