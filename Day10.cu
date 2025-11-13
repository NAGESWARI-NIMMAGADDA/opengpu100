
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
using namespace std;

vector<vector<float>> readCSV(const string& filename, int rows, int cols) {
    ifstream file(filename);
    vector<vector<float>> data(rows, vector<float>(cols));
    char comma;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            file >> data[i][j] >> comma;
    return data;
}

int main() {
    int q_rows = 3, q_cols = 4;
    int k_rows = 5, k_cols = 4;
    int v_rows = 5, v_cols = 6;
    int o_rows = 3, o_cols = 6;

    auto Q = readCSV("query_output.csv", q_rows, q_cols);
    auto K = readCSV("key_output.csv", k_rows, k_cols);
    auto V = readCSV("value_output.csv", v_rows, v_cols);
    auto expected_O = readCSV("output_output.csv", o_rows, o_cols);

    vector<vector<float>> scores(q_rows, vector<float>(k_rows, 0.0));
    for (int i = 0; i < q_rows; i++)
        for (int j = 0; j < k_rows; j++)
            for (int k = 0; k < q_cols; k++)
                scores[i][j] += Q[i][k] * K[j][k];
    for (int i = 0; i < q_rows; i++)
        for (int j = 0; j < k_rows; j++)
            scores[i][j] /= sqrt(q_cols);

    for (int i = 0; i < q_rows; i++) {
        float max_val = scores[i][0];
        for (int j = 1; j < k_rows; j++)
            if (scores[i][j] > max_val) max_val = scores[i][j];
        float sum_exp = 0.0;
        for (int j = 0; j < k_rows; j++) {
            scores[i][j] = exp(scores[i][j] - max_val);
            sum_exp += scores[i][j];
        }
        for (int j = 0; j < k_rows; j++)
            scores[i][j] /= sum_exp;
    }

    vector<vector<float>> computed_O(q_rows, vector<float>(v_cols, 0.0));
    for (int i = 0; i < q_rows; i++)
        for (int j = 0; j < v_cols; j++)
            for (int k = 0; k < k_rows; k++)
                computed_O[i][j] += scores[i][k] * V[k][j];

    cout << "Computed O:\n";
    for (auto &row : computed_O) {
        for (auto &val : row) cout << val << " ";
        cout << "\n";
    }
    cout << "Expected O:\n";
    for (auto &row : expected_O) {
        for (auto &val : row) cout << val << " ";
        cout << "\n";
    }
    cout << "Difference:\n";
    for (int i = 0; i < o_rows; i++) {
        for (int j = 0; j < o_cols; j++)
            cout << computed_O[i][j] - expected_O[i][j] << " ";
        cout << "\n";
    }
}
