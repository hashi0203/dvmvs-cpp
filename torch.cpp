#include "torch.h"

ifstream open_file(const string path) {
    ifstream ifs(path);
    if (!ifs.is_open()) {
        cerr << "Failed to open file: " << path << ".\n";
        exit(1);
    }
    return ifs;
}

float calculate_gain(const float a) {
    return sqrt(2.0 / (1 + pow(a, 2)));
}