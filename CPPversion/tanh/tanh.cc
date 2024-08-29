#include "tanh.ih"

Tanh::Tanh() : Activation(
    [](const vector<vector<double>>& x) {
        vector<vector<double>> result = x;
        for (auto& row : result) {
            for (auto& value : row) {
                value = tanh(value);
            }
        }
        return result;
    },
    [](const vector<vector<double>>& x) {
        vector<vector<double>> result = x;
        for (auto& row : result) {
            for (auto& value : row) {
                value = 1.0 - tanh(value) * tanh(value);
            }
        }
        return result;
    },
    "Tanh"
) {}