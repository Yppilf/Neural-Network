#ifndef DENSE_H
#define DENSE_H

#include "../connectiveLayer/connectiveLayer.h"

class Dense : public ConnectiveLayer {
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> bias;

    public:
        Dense(size_t input_size, size_t output_size, bool overrideInit = false);
        std::vector<std::vector<double>> forward(const std::vector<std::vector<double>> &input) override;
        std::vector<std::vector<double>> backward(const std::vector<std::vector<double>> &output_gradient, const double learning_rate) override;

        std::string saveLayer() const override;
        void loadLayer(const std::string &jsonStr) override;

        void setWeights(const std::vector<std::vector<double>> &new_weights);
        void setBiases(const std::vector<std::vector<double>> &new_biases);

    private:
        double generateRandomNumber();
};

#endif