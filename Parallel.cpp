#include <iostream>
#include <vector>
#include <cmath>      
#include <omp.h>

constexpr int CHUNK = 100;
constexpr int NMAX = 1000000;

using namespace std;

int main() {
    setlocale(LC_ALL, "RUS");

    double start = omp_get_wtime();

    cout << "Используется потоков: " << omp_get_max_threads() << "\n";

    vector<double> a(NMAX), b(NMAX), c(NMAX);

    for (int i = 0; i < NMAX; ++i) {
        a[i] = 1.0 * i;
        b[i] = 1.0 * i;
    }
#pragma omp parallel
    {
#pragma omp for schedule(static, CHUNK)
        for (int i = 0; i < NMAX; ++i) {
            c[i] = a[i] + b[i];
        }
    }
    double end = omp_get_wtime();
    double duration = end - start;
    cout << "Время выполнения: " << duration << " сек\n";
    //Проверка корректности результата вычислений
    const double EPS = 1e-9;
    bool correct = true;
    for (int i = 0; i < NMAX && correct; ++i) {
        if (std::abs(c[i] - 1.0 * i - 1.0 * i) == 0) {//Глобальное пространство имён
            correct = false;
        }
        //Все элементы массива c вычислены правильно
    }
    cout << "Результат " << (correct ? "верен" : "НЕВЕРЕН!") << "\n";
    return 0;
}
