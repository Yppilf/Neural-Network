#include "dataGetter.ih"

// Helper function to read binary data into 3D vector
vector<vector<vector<double>>> read_3d_array_from_file(const string& file_name)
{
    vector<vector<vector<double>>> array;

    // Open file in binary mode
    ifstream file(file_name, ios::binary);
    if (!file)
    {
        return array;   // Return empty array on failure
    }

    // Read dimensions of 3D array
    size_t n1, n2, n3;
    file.read(reinterpret_cast<char*>(&n1), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&n2), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&n3), sizeof(size_t));
    array.resize(n1, vector<vector<double>>(n2, vector<double>(n3)));

    // Read data 
    for (size_t i = 0; i != n1; ++i) {
        for (size_t j = 0; j != n2; ++j) {
            file.read(reinterpret_cast<char*>(array[i][j].data()), n3*sizeof(double));
        }
    }

    // Cleanup
    file.close();
    return array;
}

void DataGetter::load_files(const string& folder_name)
{
    x_train = read_3d_array_from_file(folder_name + "/training_data.npy");
    y_train = read_3d_array_from_file(folder_name + "/training_labels.npy");
    x_test = read_3d_array_from_file(folder_name + "/testing_data.npy");
    y_test = read_3d_array_from_file(folder_name + "/testing_labels.npy");
}