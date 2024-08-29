#include "activation.ih"

void Activation::loadLayer(const string &jsonStr) {
    this->type = extractJsonField(jsonStr, "type");
}