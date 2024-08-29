#ifndef HELPER_H
#define HELPER_H

#include <string>
#include <vector>

std::string serializeMatrix(const std::vector<std::vector<double>> &matrix);
std::vector<std::vector<double>> deserializeMatrix(const std::string &matrixStr);
std::string extractJsonField(const std::string &jsonStr, const std::string &fieldName);

#endif