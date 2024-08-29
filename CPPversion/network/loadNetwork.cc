#include "network.ih"

void setLayerWeightsAndBiases(const shared_ptr<ConnectiveLayer> &layerObj, const string &layerStr) {
    string weightsStr = extractJsonField(layerStr, "weights");
    string biasStr = extractJsonField(layerStr, "bias");

    vector<vector<double>> weights = deserializeMatrix(weightsStr);
    vector<vector<double>> biases = deserializeMatrix(biasStr);

    layerObj->setWeights(weights);
    layerObj->setBiases(biases);
}

void Network::loadNetwork(const string &filename){
    ifstream file(filename);
    if (!file) {
        cerr << "Unable to open file for writing: " << filename << "\n";
        return;
    }

    string jsonContent((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();

    learning_rate = stod(extractJsonField(jsonContent, "learning_rate"));
    epochs = stoul(extractJsonField(jsonContent, "epochs"));
    string layersContent = extractJsonField(jsonContent, "layers");
    istringstream iss(layersContent);
    string layerStr;

    network.clear();
    while (getline(iss, layerStr, '}')) {
        if (layerStr.empty() || layerStr == "," || layerStr == "[") {
            continue;
        }
        layerStr += "}";

        shared_ptr<Layer> layerObj;

        // Add other layer types here
        // TODO could this be cleaner with using extractJsonField for finding type
        if (layerStr.find("\"type\":\"Dense\"") != string::npos) {
            layerObj = make_shared<Dense>(1,1,true);
        } else if (layerStr.find("\"type\":\"Tanh\"") != string::npos) {
            layerObj = make_shared<Tanh>();
        } else if (layerStr.find("\"type\":\"Sigmoid\"") != string::npos) {
            layerObj = make_shared<Sigmoid>();
        }

        // For scalability if other types of connective layers would get added
        if (auto connectiveLayer = dynamic_pointer_cast<ConnectiveLayer>(layerObj)) {
            setLayerWeightsAndBiases(connectiveLayer, layerStr);
        } 

        if (layerObj) {
            network.push_back(layerObj);
        }
    }
}