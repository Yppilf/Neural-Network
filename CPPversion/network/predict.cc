#include "network.ih"

vector<vector<double>> Network::predict(const vector<vector<double>>& input) 
{
    vector<vector<double>> output = input;
    for (const auto& layer : network) {
        output = layer->forward(output);
    }
    return output;
}