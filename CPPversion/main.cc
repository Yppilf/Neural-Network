#include "main.ih"

int main()
{
    // Acquire data
    DataGetter dataGetter;
    dataGetter.load_files("mini_mnist_data");
    auto x_train = dataGetter.getXTrain();
    auto y_train = dataGetter.getYTrain();
    auto x_test = dataGetter.getXTest();
    auto y_test = dataGetter.getYTest();

    // Create network

    // Train network

    // Save network
}