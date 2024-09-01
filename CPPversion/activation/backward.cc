#include "activation.ih"
#include <iostream>

vector<vector<double>> Activation::backward(const vector<vector<double>>& /*output_gradient*/, const double /*learning_rate*/) {
    return activation_prime(this->input);
}