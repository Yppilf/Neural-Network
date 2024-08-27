#include "dataGetter.ih"

// Helper function to read binary data into 3D vector
vector<vector<vector<double>>> read_3d_array_from_csv(const string& filename)
{
    vector<vector<vector<double>>> array;
    string line;
    vector<vector<double>> slice;

    ifstream file(filename);
    if (!file) {
        return array;   // Return empty array
    }

    while(getline(file,line)) {
        if (line.empty()) {
            if (!slice.empty()) {
                array.push_back(slice);
                slice.clear();
            }
            continue;
        }

        vector<double> row;
        stringstream ss(line);
        string item;

        while (getline(ss, item, ',')) {
            row.push_back(stod(item));
        }

        slice.push_back(row);
    }
    
    if (!slice.empty()) {
        array.push_back(slice);
    }

    return array;
}

void DataGetter::load_files(const string& folder_name)
{
    x_train = read_3d_array_from_csv(folder_name + "/training_data.npy");
    y_train = read_3d_array_from_csv(folder_name + "/training_labels.npy");
    x_test = read_3d_array_from_csv(folder_name + "/testing_data.npy");
    y_test = read_3d_array_from_csv(folder_name + "/testing_labels.npy");
}