#include "activation.ih"

string Activation::saveLayer() const {
    ostringstream oss;
    oss << "{";
    oss << "\"type\":\"" << type << "\"";
    oss << "}";
    return oss.str();
}