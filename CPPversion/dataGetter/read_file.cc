#include "dataGetter.ih"

vector<vector<vector<double>>> DataGetter::read_file(const string &filename)
{
    vector<vector<vector<double>>> array;
    ifstream file(filename);

    string line;
    while (getline(file, line)) {
        vector<vector<double>> sample;
        stringstream ss(line);
        string value;

        while (getline(ss, value, ',')) {
            sample.push_back({stod(value)});
        }

        array.push_back(sample);
    }

    file.close();
    return array;
}