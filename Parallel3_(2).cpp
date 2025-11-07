#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <random>

using namespace std;

const double eps = 1e-9;

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
        A[i][i] = sum_abs + 1.0 + 0.5 * abs(off_diag(gen)); // небольшой запас

        // Сгенерируем b так, чтобы решение было x = [1, 1, ..., 1]
        // Тогда b[i] = sum_j A[i][j] * 1 = сумма строки A[i]
        b[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            b[i] += A[i][j];
        }
    }
}

// Метод Якоби
void JacobiParallel(int maxIter, vector<vector<double>>& A, vector<double>& b, vector<double>& x) {
    int size = b.size();
    const double eps = 1e-9;
    vector<double> x_k1(size);
    int iter = 0;
    double norm;
    double start = omp_get_wtime();

    do {
#pragma omp parallel
        {
            // Параллельно вычисляем новые значения
#pragma omp for
            for (int i = 0; i < size; i++) {
                double sum = b[i];
                for (int j = 0; j < size; j++) {
                    if (i != j)
                        sum -= A[i][j] * x[j];
                }
                x_k1[i] = sum / A[i][i];
            }
        } // конец параллельного региона

        // Последовательный расчёт нормы и обновление x
        norm = abs(x[0] - x_k1[0]);
#pragma omp for
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
    cout << "Время выполнения (Якоби): " << duration << " секунд\n";
    cout << "Метод сошёлся за " << iter << " итераций.\n";
}

// Метод Зейделя
void SeidelParallel(int maxIter, const vector<vector<double>>& A, const vector<double>& b, vector<double>& x) {
    int n = static_cast<int>(b.size());
    int iter = 0;
    double norm;
    double start = omp_get_wtime();
    do {
        norm = 0.0;
        for (int i = 0; i < n; ++i) {
            double sum = b[i];
#pragma omp parallel
            {
                double local_sum = 0.0;
                // Каждый поток вычисляет свою часть суммы
#pragma omp for nowait // Отключение барьерной синхронизации - позволяет потокам не ждать друг друга в конце параллельного региона.
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        local_sum += A[i][j] * x[j];//x[j] - хранятся и новые и старые x
                    }
                }
                // Собираем все частичные суммы
#pragma omp atomic // Обеспечивает атомарность (неразрывность) для простых операций над одной переменной.
                sum -= local_sum;
            }

            double x_new = sum / A[i][i];
            double diff = fabs(x_new - x[i]);

            // Обновляем норму
#pragma omp critical // Гарантирует, что только один поток может выполнять этот блок кода в данный момент времени.
            {
                if (diff > norm) {
                    norm = diff;
                }
            }
            x[i] = x_new;
        }

        iter++;
        if (iter >= maxIter) {
            cout << "Предупреждение: метод Зейделя не сошёлся за " << maxIter << " итераций.\n";
            break;
        }
    } while (norm > eps);

    double end = omp_get_wtime();
    double duration = end - start;
    cout << "Время выполнения (Зейдель): " << duration << " секунд\n";
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
    cout << "Введите размер матрицы: ";
    cin >> size;
    cout << "\n";
    
    generateDiagonallyDominantSystem(size, A, b);
    vector<double> x(size, 0.0);

    int maxIter;
    cout << "Введите максимальное число итераций: ";
    cin >> maxIter;
    cout << "\n";

    // Устанавливаем количество потоков
    int num_threads;
    cout << "Введите количество потоков: ";
    cin >> num_threads;
    omp_set_num_threads(num_threads);

    cout << "=== Метод Якоби ===" << endl;
    JacobiParallel(maxIter, A, b, x);

    // Сбрасываем решение для следующего метода
    fill(x.begin(), x.end(), 0.0);

    cout << "\n=== Метод Зейделя ===" << endl;
    SeidelParallel(maxIter, A, b, x);

    return 0;
}