#include "dense.ih"

vector<vector<double>> Dense::backward(const vector<vector<double>> &output_gradient, const double learning_rate) {
    size_t rows = weights.size();
    size_t cols = input.size();

    vector<vector<double>> weights_gradient(rows, vector<double>(cols));
    vector<vector<double>> input_gradient(cols, vector<double>(output_gradient[0].size()));

    // weights gradient is dot product between output gradient and transmute of input
    for (size_t i = 0; i != rows; ++i) {
        for (size_t j = 0; j != cols; ++j) {
            weights_gradient[i][j] = 0.0;
            for (size_t k = 0; k != output_gradient[0].size(); ++k) {
                weights_gradient[i][j] += output_gradient[i][k] * input[j][k]; 
            }
        }
    }

    // input gradient is dot product between transmute of weights and output gradient
    for (size_t i = 0; i != cols; ++i) {
        for (size_t j = 0; j != output_gradient[0].size(); ++j) {
            input_gradient[i][j] = 0.0;
            for (size_t k = 0; k != rows; ++k) {
                input_gradient[i][j] += weights[k][i] * output_gradient[k][j];
            }
        }
    }

    for (size_t i = 0; i != rows; ++i) {
        for (size_t j = 0; j != cols; ++j) {
            weights[i][j] -= learning_rate * weights_gradient[i][j];
        }
        for (size_t j = 0; j != output_gradient[0].size(); ++j) {
            bias[i][0] -= learning_rate * output_gradient[i][j];
        }
    }
    return input_gradient;
}