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

            cout << "train: Backward propagate...\n";
            // Backward propagate
            vector<vector<double>> grad = loss_prime(y_train[i], output);
            cout << "train: Acquired losses...\n";
            for (auto it = network.rbegin(); it != network.rend(); ++it) {
                cout << "Iterating...\n";
                cout << "grad dimensions: " << grad.size() << "x" << grad[0].size() << "\n";
                grad = (*it)->backward(grad, learning_rate);
            }
        }
        cout << "train: calculate errors...\n";
        error /= x_train.size();
        errors.push_back(error);

        if (verbose) {
            cout << (e+1) << "/" << epochs << ", error=" << error << "\n";
        }
    }
    return errors;
}