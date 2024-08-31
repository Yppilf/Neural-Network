#ifndef LOSSES_H
#define LOSSES_H

#include <vector>
#include <functional>

extern std::function<double(const std::vector<std::vector<double>>& y_true, const std::vector<std::vector<double>>& y_pred)> mse;
extern std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>& y_true, const std::vector<std::vector<double>>& y_pred)> mse_prime;

#endif