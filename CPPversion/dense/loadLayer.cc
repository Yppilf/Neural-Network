#include "dense.ih"

void Dense::loadLayer(const string &jsonStr) {
    string weightsStr = extractJsonField(jsonStr, "weights");
    string biasStr = extractJsonField(jsonStr, "bias");

    weights = deserializeMatrix(weightsStr);
    bias = deserializeMatrix(biasStr);
}