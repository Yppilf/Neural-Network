#include "network.ih"

vector<double> Network::train(
    function<double(const vector<vector<double>>&, const vector<vector<double>>&)>& loss,
    function<vector<vector<double>>(const vector<vector<double>>&, const vector<vector<double>>&)>& loss_prime,
    const vector<vector<vector<double>>>& x_train,
    const vector<vector<vector<double>>>& y_train,
    size_t epochs, bool verbose
) {
    errors.clear();
    this->epochs = epochs;

    for (size_t e = 0; e != epochs; ++e) {
        double error = 0;
        for (size_t i = 0; i != x_train.size(); ++i) {
            // Forward propagate
            vector<vector<double>> output = predict(x_train[i]);

            // error
            error += loss(y_train[i], output);

            // Backward propagate
            vector<vector<double>> grad = loss_prime(y_train[i], output);
            for (auto it = network.rbegin(); it != network.rend(); ++it) {
                grad = (*it)->backward(grad, learning_rate);
            }
        }
        error /= x_train.size();
        errors.push_back(error);

        if (verbose) {
            cout << (e+1) << "/" << epochs << ", error=" << error << "\n";
        }
    }
    return errors;
}