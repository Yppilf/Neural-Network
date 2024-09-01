#include "main.ih"

int main()
{
    cout << "1";
    // Acquire data
    DataGetter dataGetter;
    dataGetter.load_files("../mnist_data_2");
    auto x_train = dataGetter.getXTrain();
    auto y_train = dataGetter.getYTrain();
    auto x_test = dataGetter.getXTest();
    auto y_test = dataGetter.getYTest();

    cout << "2";
    // Transpose data where needed
    y_train = dataGetter.transposeLabels(y_train);
    y_test = dataGetter.transposeLabels(y_test);

    cout << "3";
    // Create network
    vector<shared_ptr<Layer>> networkStructure = {
        make_shared<Dense>(28*28, 40),
        make_shared<Tanh>(),
        make_shared<Dense>(40,10),
        make_shared<Tanh>()
    };
    Network network(networkStructure, 0.1);

    cout << "4";
    // Train network
    vector<double> errors = network.train(mse, mse_prime, x_train, y_train, 50, true);

    cout << "5";
    // Save network
    network.saveNetwork("mnist3.json");
}