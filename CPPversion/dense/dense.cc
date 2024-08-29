#include "dense.ih"

Dense::Dense(size_t input_size, size_t output_size, bool overrideInit) {
    if (!overrideInit) {
        weights.resize(output_size, vector<double>(input_size));
        bias.resize(output_size, vector<double>(1));
        for (size_t i = 0; i != output_size; ++i) {
            for (size_t j = 0; j != input_size; ++j) {
                weights[i][j] = generateRandomNumber();
            }
            bias[i][0] = generateRandomNumber();
        }
    }
}