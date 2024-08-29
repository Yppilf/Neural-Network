#include "dense.ih"

vector<vector<double>> Dense::forward(const vector<vector<double>> &input) {
    this->input = input;
    size_t rows = weights.size();
    size_t cols = input[0].size();
    output.resize(rows, vector<double>(cols));

    for (size_t i = 0; i != rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double sum;
            for (size_t k = 0; k != weights[0].size(); ++k) {
                sum += weights[i][k] * input[k][j];
            }
            output[i][j] = sum + bias[i][0];
        }
    }
    return output;
}