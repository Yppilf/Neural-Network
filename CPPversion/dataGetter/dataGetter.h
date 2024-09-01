#ifndef DATAGETTER_H
#define DATAGETTER_H

#include <vector>
#include <string>

class DataGetter
{
    
    std::vector<std::vector<std::vector<double>>> x_train;
    std::vector<std::vector<std::vector<double>>> y_train;
    std::vector<std::vector<std::vector<double>>> x_test;
    std::vector<std::vector<std::vector<double>>> y_test;

    public:
        // Constructor
        DataGetter();

        // Getter methods for each attribute
        std::vector<std::vector<std::vector<double>>> getXTrain() const;
        std::vector<std::vector<std::vector<double>>> getXTest() const;
        std::vector<std::vector<std::vector<double>>> getYTrain() const;
        std::vector<std::vector<std::vector<double>>> getYTest() const;

        // Loading data from files
        void load_files(const std::string& folder_name);

        // transpose i,j,k to j,i,k
        std::vector<std::vector<std::vector<double>>> transposeLabels(const std::vector<std::vector<std::vector<double>>> &data);
};

#endif