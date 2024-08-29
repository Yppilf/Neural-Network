#ifndef CONNECTIVELAYER_H
#define CONNECTIVELAYER_H

#include "../layer/layer.h"

class ConnectiveLayer : public Layer {
    public:
        virtual void setWeights(const std::vector<std::vector<double>> &new_weights) = 0;
        virtual void setBiases(const std::vector<std::vector<double>> &new_biases) = 0;
};

#endif