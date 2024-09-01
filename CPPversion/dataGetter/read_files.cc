#include "dataGetter.ih"

void DataGetter::read_files(const string &folder_name)
{
    x_train = read_file(folder_name + "/x_train.csv");
    y_train = read_file(folder_name + "/y_train.csv");
    x_test = read_file(folder_name + "/x_test.csv");
    y_test = read_file(folder_name + "/y_test.csv");
}