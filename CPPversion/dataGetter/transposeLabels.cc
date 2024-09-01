#include "dataGetter.ih"

vector<vector<vector<double>>> DataGetter::transposeLabels(const vector<vector<vector<double>>> &data) {
    if (data.empty()) return {};

    size_t num_samples = data.size();
    size_t rows = data[0].size();
    size_t cols = data[0][0].size();

    vector<vector<vector<double>>> transposed(rows, vector<vector<double>>(num_samples, vector<double>(cols)));

    for (size_t i = 0; i != num_samples; ++i) {
        for (size_t j = 0; j != rows; ++j) {
            for (size_t k = 0; k < cols; ++k) {
                transposed[j][i][k] = data[i][j][k];
            }
        }
    }
    return transposed;
}