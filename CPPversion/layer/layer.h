#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <string>
#include "../helper/helper.h"

class Layer
{
    public:
        virtual std::vector<std::vector<double>> forward(const std::vector<std::vector<double>> &input) = 0;
        virtual std::vector<std::vector<double>> backward(const std::vector<std::vector<double>> &output_gradient, double learning_rate) = 0;

        virtual std::string saveLayer() const = 0;
        virtual void loadLayer(const std::string &jsonStr) = 0;
    protected:
        std::vector<std::vector<double>> input;
        std::vector<std::vector<double>> output;
};

#endif