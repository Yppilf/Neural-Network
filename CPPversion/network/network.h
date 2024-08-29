#ifndef NETWORK_H
#define NETWORK_H

#include <memory>
#include <functional>
#include "../layer/layer.h"

class Network 
{
    std::vector<std::shared_ptr<Layer>> network;
    double learning_rate;
    size_t epochs;
    std::vector<double> errors;

    public:
        Network(const std::vector<std::shared_ptr<Layer>>& layers,
        double learning_rate = 0.01,
        bool overrideInit = false);

        std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& input);
        std::vector<double> train(
            std::function<double(const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&)>& loss,
            std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&)>& loss_prime,
            const std::vector<std::vector<std::vector<double>>>& x_train,
            const std::vector<std::vector<std::vector<double>>>& y_train,
            size_t epochs = 1000, 
            bool verbose = true
        );

        void saveNetwork(const std::string &filename) const;
        void loadNetwork(const std::string &filename);
};

#endif