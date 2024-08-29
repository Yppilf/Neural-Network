#include "sigmoid.ih"

Sigmoid::Sigmoid() : Activation(
    [](const vector<vector<double>>& x) {
        vector<vector<double>> result = x;
        for (auto& row : result) {
            for (auto& value : row) {
                value = 1.0 / (1.0 + exp(-value));
            }
        }
        return result;
    },
    [](const vector<vector<double>>& x) {
        vector<vector<double>> result = x;
        for (auto& row : result) {
            for (auto& value : row) {
                double s = 1.0 / (1.0 + exp(-value));
                value = s * (1-s);
            }
        }
        return result;
    },
    "Sigmoid"
) {}