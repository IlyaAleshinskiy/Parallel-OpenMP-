#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>  
#include <iomanip>
#include <random>

using namespace std;

// Генерирует строго диагонально доминирующую матрицу A и вектор b
void generateDiagonallyDominantSystem(int N, vector<vector<double>>& A, vector<double>& b) {
    A.assign(N, vector<double>(N, 0.0));
    b.assign(N, 0.0);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> off_diag(-1.0, 1.0);  // недиагональные элементы

    for (int i = 0; i < N; ++i) {
        double sum_abs = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            A[i][j] = off_diag(gen);
            sum_abs += abs(A[i][j]);
        }
        // Сделаем диагональный элемент строго больше суммы остальных
        A[i][i] = sum_abs + 2.0; // небольшой запас

        // Сгенерируем b так, чтобы решение было x = [1, 1, ..., 1]
        // Тогда b[i] = sum_j A[i][j] * 1 = сумма строки A[i]
        b[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            b[i] += A[i][j];
        }
    }
}

void JacobiSerial(int maxIter, vector<vector<double>>& A, vector<double>& b, vector<double>& x) {
    int size = b.size();
    const double eps = 1e-9;
    vector<double> x_k1(size);
    int iter = 0;
    double norm;
    double start = omp_get_wtime();
    do {
        for (int i = 0; i < size; i++) {
            double sum = b[i];
            for (int j = 0; j < size; j++) {
                if (i != j)
                    sum -= A[i][j] * x[j];
            }
            x_k1[i] = sum / A[i][i];
        }

        norm = abs(x[0] - x_k1[0]);
        for (int i = 0; i < size; ++i) {
            double diff = abs(x[i] - x_k1[i]);
            if (diff > norm)
                norm = diff;
            x[i] = x_k1[i];
        }
        iter++;
        if (iter > maxIter) {
            cout << "Предупреждение: метод не сошёлся за " << maxIter << " итераций.\n";
            break;
        }
    } while (norm > eps);
    double end = omp_get_wtime();
    double duration = end - start;
    cout << "Время выполнения (последовательно): " << duration << " нс\n";
    cout << "Метод сошёлся за " << iter << " итераций.\n";
}

void JacobiParallel(int maxIter, vector<vector<double>>& A, vector<double>& b, vector<double>& x) {
    int size = b.size();
    const double eps = 1e-9;
    vector<double> x_k1(size);
    int iter = 0;
    double norm;
    double start = omp_get_wtime();
    do {
    #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            double sum = b[i];
            for (int j = 0; j < size; j++) {
                if (i != j)
                    sum -= A[i][j] * x[j];
            }
            x_k1[i] = sum / A[i][i];
        }
        // Последовательный расчёт нормы и обновление x
        norm = abs(x[0] - x_k1[0]);
#pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            double diff = abs(x[i] - x_k1[i]);
            if (diff > norm)
                norm = diff;
            x[i] = x_k1[i];
        }
        iter++;
        if (iter > maxIter) {
            cout << "Предупреждение: метод не сошёлся за " << maxIter << " итераций.\n";
            break;
        }
    } while (norm > eps);
    double end = omp_get_wtime();
    double duration = end - start;
    cout << "Время выполнения (параллельно): " << duration << " нс\n";
    cout << "Метод сошёлся за " << iter << " итераций.\n";
}

int main() {
    setlocale(LC_ALL, "Russian");
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr); cout.tie(nullptr);
    cout << fixed << setprecision(8);
    vector<vector<double>> A;
    vector<double> b;
    int size;
    cout << "Введите размер матрицы: "; cin >> size; cout << "\n";
    omp_set_num_threads(4);
    generateDiagonallyDominantSystem(size, A, b);
    vector<double> x(size, 0.0);

    int maxIter;
    cout << "Введите максимальное число итераций: ";
    cin >> maxIter;
    cout << "\n";

    JacobiSerial(maxIter, A, b, x);
    for (int i = 0; i < b.size(); ++i) {
        x[i] = 0;
    }
    cout << "\n";
    JacobiParallel(maxIter, A, b, x);
    return 0;
}