#include "activation.ih"
#include <iostream>

vector<vector<double>> Activation::backward(const vector<vector<double>>& /*output_gradient*/, const double /*learning_rate*/) {
    cout << "tanh: backward propagating...\n";
    return activation_prime(this->input);
}