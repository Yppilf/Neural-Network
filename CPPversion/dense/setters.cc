#include "dense.ih"

void Dense::setWeights(const vector<vector<double>> &new_weights) {
    this->weights = new_weights;
}

void Dense::setBiases(const vector<vector<double>> &new_biases) {
    this->bias = new_biases;
}