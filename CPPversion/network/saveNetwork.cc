#include "network.ih"

void Network::saveNetwork(const string &filename) const 
{
    ofstream file(filename);

    if (!file) {
        cerr << "Unable to open file for writing: " << filename << "\n";
        return;
    }

    file << "{";
    file << "\"learning_rate\":" << learning_rate << ",";
    file << "\"epochs\":" << epochs << ",";
    file << "\"layers\": [";

    for (size_t i = 0; i != network.size(); ++i) {
        file << network[i]->saveLayer();
        if (i < network.size() - 1) {
            file << ",";
        }
    }
    file << "]}";
    file.close();
}