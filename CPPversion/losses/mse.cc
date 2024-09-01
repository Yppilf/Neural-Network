#include "losses.ih"
#include <iostream>

function<double(const vector<vector<double>>&, const vector<vector<double>>&)> mse =
    [](const vector<vector<double>>& y_true, const vector<vector<double>>& y_pred) -> double {
        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            for (size_t j = 0; j < y_true[i].size(); ++j) {
                sum += pow(y_true[i][j] - y_pred[i][j], 2);
            }
        }
        return sum / y_true.size();
    };

function<vector<vector<double>>(const vector<vector<double>>&, const vector<vector<double>>&)> mse_prime =
    [](const vector<vector<double>>& y_true, const vector<vector<double>>& y_pred) -> vector<vector<double>> {
        cout << "mse: Determining loss gradient...\n";
        cout << "mse: y_true size: " << y_true.size() << "x" << y_true[0].size() << "\n";
        cout << "mse: y_pred size: " << y_pred.size() << "x" << y_pred[0].size() << "\n";
        vector<vector<double>> gradient(y_true.size(), vector<double>(y_true[0].size()));
        for (size_t i = 0; i < y_true.size(); ++i) {
            for (size_t j = 0; j < y_true[i].size(); ++j) {
                gradient[i][j] = 2 * (y_pred[i][j] - y_true[i][j]) / y_true.size();
            }
        }
        return gradient;
    };

