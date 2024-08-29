#include "layer.ih"

vector<vector<double>> Layer::deserializeMatrix(const string &matrixStr) {
    vector<vector<double>> matrix;
    istringstream iss(matrixStr);
    string rowStr;

    while (getline(iss, rowStr, ']')) {
        if (rowStr.empty() || rowStr == ",") {
            continue;
        }
        istringstream rowStream(rowStr.substr(1));    // Skip opening '[' 
        vector<double> row;
        string numStr;
        while (getline(rowStream, numStr, ',')) {
            row.push_back(stod(numStr));
        }
        matrix.push_back(row);
    }
    return matrix;
}