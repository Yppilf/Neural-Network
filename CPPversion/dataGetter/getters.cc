#include "dataGetter.ih"

vector<vector<vector<double>>> DataGetter::getXTrain() const 
{
    return x_train;
}

vector<vector<vector<double>>> DataGetter::getYTrain() const 
{
    return y_train;
}

vector<vector<vector<double>>> DataGetter::getXTest() const 
{
    return x_test;
}

vector<vector<vector<double>>> DataGetter::getYTest() const 
{
    return y_test;
}