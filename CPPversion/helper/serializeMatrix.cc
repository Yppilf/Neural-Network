#include "helper.ih"

string serializeMatrix(const std::vector<std::vector<double>> &matrix)
{
    ostringstream oss;
    for (size_t i =0; i != matrix.size(); ++i) {
        oss << "[";
        for (size_t j = 0; j != matrix[i].size(); ++j) {
            oss << matrix[i][j];
            if (j < matrix[i].size() - 1) {
                oss << ",";
            }
        }
        oss << "]";
        if (i < matrix.size() - 1) {
            oss << ",";
        }
    }
    return oss.str();    
}