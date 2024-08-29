#include "activation.ih"

Activation::Activation(function<vector<vector<double>>(const vector<vector<double>>&)> activation,
    function<vector<vector<double>>(const vector<vector<double>>&)> activation_prime,
    const string &type
) : activation(activation), activation_prime(activation_prime), type(type) {}