#include "dense.ih"

double Dense::generateRandomNumber() {
    static random_device rd;
    static mt19937 gen(rd());
    static normal_distribution<> d(0,1);
    return d(gen);
}