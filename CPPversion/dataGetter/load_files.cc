#include "dataGetter.ih"

// Helper function to read binary data into 3D vector
vector<vector<vector<double>>> read_3d_array_from_csv(const string& filename)
{
    vector<vector<vector<double>>> array;
    string line;
    vector<vector<double>> slice;
    vector<double> row;

    ifstream file(filename);
    if (!file) {
        cerr << "Couldnt open file" << filename << "\n";
        return array;   // Return empty array
    }

    while (getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        if (line.empty()) {
            if (!row.empty() || !slice.empty()) {
                if (!row.empty()) {
                    slice.push_back(row);
                    row.clear();
                }
                if (!slice.empty()) {
                    array.push_back(slice);
                    slice.clear();
                }
            }
        } else {
            stringstream ss(line);
            string item;
            while (getline(ss, item, ',')) {
                row.push_back(stod(item));
            }
            if (!row.empty()) {
                slice.push_back(row);
            }
        }
    }

    if (!row.empty()) {
        slice.push_back(row);
    }
    if (!slice.empty()) {
        array.push_back(slice);
    }

    return array;
}

void DataGetter::load_files(const string& folder_name)
{
    x_train = read_3d_array_from_csv(folder_name + "/x_train.csv");
    y_train = read_3d_array_from_csv(folder_name + "/y_train.csv");
    x_test = read_3d_array_from_csv(folder_name + "/x_test.csv");
    y_test = read_3d_array_from_csv(folder_name + "/y_test.csv");
}