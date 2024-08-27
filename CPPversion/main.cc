#include "main.ih"

int main()
{
    // Acquire data
    DataGetter dataGetter;
    dataGetter.load_files("../mini_mnist_data_2");
    auto x_train = dataGetter.getXTrain();
    auto y_train = dataGetter.getYTrain();
    auto x_test = dataGetter.getXTest();
    auto y_test = dataGetter.getYTest();

    cout << y_test[0][0][0] << "\n";

    // Create network

    // Train network

    // Save network
}