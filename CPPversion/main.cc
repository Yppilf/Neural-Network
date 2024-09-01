#include "main.ih"

void print_3d_vector(const vector<vector<vector<double>>>& vec) 
{
    for (size_t i = 0; i < vec.size(); ++i) {
        cout << "Slice " << i << ":\n";
        for (size_t j = 0; j < vec[i].size(); ++j) {
            for (size_t k = 0; k < vec[i][j].size(); ++k) {
                cout << vec[i][j][k] << " ";
            }
            cout << "\n";
        }
        cout << "\n";
    }
}

int main()
{
    // Acquire data
    DataGetter dataGetter;
    dataGetter.load_files("../mini_mnist_data_2");
    auto x_train = dataGetter.getXTrain();
    auto y_train = dataGetter.getYTrain();
    auto x_test = dataGetter.getXTest();
    auto y_test = dataGetter.getYTest();

    // Transpose data where needed
    y_train = dataGetter.transposeLabels(y_train);
    y_test = dataGetter.transposeLabels(y_test);

    // Create network
    vector<shared_ptr<Layer>> networkStructure = {
        make_shared<Dense>(28*28, 40),
        make_shared<Tanh>(),
        make_shared<Dense>(40,10),
        make_shared<Tanh>()
    };
    Network network(networkStructure, 0.1);

    cout << "Training network...\n";
    // Train network
    cout << "y_train shape: " << y_train.size() << "x" << y_train[0].size() << "x" << y_train[0][0].size() << "\n";
    vector<double> errors = network.train(mse, mse_prime, x_train, y_train, 10, true);

    cout << "Saving network...\n";
    // Save network
    network.saveNetwork("mnist3.json");

    cout << "Done\n";
}