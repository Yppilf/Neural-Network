#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../layer/layer.h"
#include <functional>

class Activation : public Layer {
    public:
        Activation(std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&)> activation,
            std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&)> activation_prime,
            const std::string &type
        );

        std::vector<std::vector<double>> forward(const std::vector<std::vector<double>> &input) override;
        std::vector<std::vector<double>> backward(const std::vector<std::vector<double>> &output_gradient, const double learning_rate) override;

        std::string saveLayer() const override;
        void loadLayer(const std::string &jsonStr) override;

    protected:
        std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&)> activation;
        std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&)> activation_prime;
        std::string type;
};

#endif