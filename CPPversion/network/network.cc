#include "network.ih"

Network::Network(const vector<shared_ptr<Layer>>& layers, double learning_rate, bool overrideInit)
    : learning_rate(learning_rate), epochs(0) 
{
    if (!overrideInit) {
        this->network = layers;
    }
}