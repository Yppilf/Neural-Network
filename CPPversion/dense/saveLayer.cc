#include "dense.ih"

string Dense::saveLayer() const {
    ostringstream oss;
    oss << "{";
    oss << "\"weights\":[" << serializeMatrix(weights) << "],";
    oss << "\"bias\":[" << serializeMatrix(bias) << "],";
    oss << "\"type\":\"Dense\"";
    oss << "}";
    return oss.str();
}