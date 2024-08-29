#include "activation.ih"

vector<vector<double>> Activation::forward(const vector<vector<double>> &input) {
    this->input = input;
    return activation(this->input);
}