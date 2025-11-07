#include <vector>
#include <iostream>
#include <omp.h>

using namespace std;

void sumRowsDivisibleByThreadId(int m, int n, const vector<vector<double>>& A, vector<vector<double>>& results) {
    int p = omp_get_max_threads(); // или задать явно
    results.assign(p, vector<double>(n, 0.0));
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int i = tid + 1; // логический делитель (начинаем с 1)

        // Каждый поток заполняет results[tid]
        vector<double>& sum_vec = results[tid];

        // Проходим по строкам в 1-based нумерации: r1 = 1..m
        for (int r1 = 1; r1 <= m; ++r1) {
            if (r1 % i == 0) { // r1 делится на i
                int r0 = r1 - 1; // 0-based индекс строки
                for (int j = 0; j < n; ++j) {
                    sum_vec[j] += A[r0][j];
                }
            }
        }
    }
}
int main() {
    setlocale(LC_ALL, "Russian");
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr); cout.tie(nullptr);
    vector<vector<double>> A;

    sumRowsDivisibleByThreadId(m, n, A, results);
    return 0;
}
