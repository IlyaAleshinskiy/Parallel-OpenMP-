#include <iostream>
#include <iomanip>
#include <omp.h>
#include <cmath>

using namespace std;

// Вспомогательная функция: выделение матрицы
double** AllocateMatrix(int size) {
    double** matrix = new double* [size + 1];
    for (int i = 0; i <= size; ++i) {
        matrix[i] = new double[size + 1]();
    }
    return matrix;
}

void FreeMatrix(double** matrix, int size) {
    for (int i = 0; i <= size; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void ProcessInit(double**& matrixA, double**& matrixB, double**& result, int& size) {
    cout << "Введите размер квадратных матриц (n): ";
    cin >> size;

    matrixA = AllocateMatrix(size);
    matrixB = AllocateMatrix(size);
    result = AllocateMatrix(size);

    cout << "Введите элементы первой матрицы A (" << size << "x" << size << "):\n";
    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            cout << "A[" << i << "][" << j << "] = ";
            cin >> matrixA[i][j];
        }
    }

    cout << "Введите элементы второй матрицы B (" << size << "x" << size << "):\n";
    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            cout << "B[" << i << "][" << j << "] = ";
            cin >> matrixB[i][j];
        }
    }

    cout << "Инициализация завершена.\n\n";
}

void SerialProduct(double** matrixA, double** matrixB, double** result, int size) {
    double start = omp_get_wtime();

    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            result[i][j] = 0.0;
            for (int k = 1; k <= size; ++k) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }

    double end = omp_get_wtime();
    double duration = end - start;

    cout << "Результат последовательного умножения:\n";
    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            cout << fixed << setprecision(8) << result[i][j] << "\t";
        }
        cout << "\n";
    }
    cout << "Время выполнения (последовательно): " << duration << " нс\n\n";
}

void ParallelProduct_Row(double** matrixA, double** matrixB, double** result, int size) {
    double start = omp_get_wtime();

#pragma omp parallel for
    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            double sum = 0.0;
            for (int k = 1; k <= size; ++k) {
                sum += matrixA[i][k] * matrixB[k][j];
            }
            result[i][j] = sum;
        }
    }

    double end = omp_get_wtime();
    double duration = end - start;

    cout << "Результат параллельного умножения (по строкам):\n";
    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            cout << fixed << setprecision(8) << result[i][j] << "\t";
        }
        cout << "\n";
    }
    cout << "Время выполнения (по строкам): " << duration << " нс\n\n";
}

void ParallelProduct_Column(double** matrixA, double** matrixB, double** result, int size) {
    double start = omp_get_wtime();

    // Обнуляем результат
    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            result[i][j] = 0.0;
        }
    }

#pragma omp parallel for
    for (int j = 1; j <= size; ++j) {
        for (int i = 1; i <= size; ++i) {
            double sum = 0.0;
            for (int k = 1; k <= size; ++k) {
                sum += matrixA[i][k] * matrixB[k][j];
            }
#pragma omp atomic
            result[i][j] += sum;
        }
    }

    double end = omp_get_wtime();
    double duration = end - start;

    cout << "Результат умножения (по столбцам):\n";
    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            cout << fixed << setprecision(8) << result[i][j] << "\t";
        }
        cout << "\n";
    }
    cout << "Время выполнения (по столбцам): " << duration << " нс\n\n";
}

void ParallelProduct_Block(double** matrixA, double** matrixB, double** result, int size) {
    double start = omp_get_wtime();

    int GridThreadsNum = 4;
    int GridSize = static_cast<int>(sqrt(static_cast<double>(GridThreadsNum)));
    int BlockSize = size / GridSize;

    if (size % GridSize != 0) {
        cerr << "Ошибка: размер должен делиться на " << GridSize << "!\n";
        return;
    }

    // Обнуляем результат
    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            result[i][j] = 0.0;
        }
    }

    omp_set_num_threads(GridThreadsNum);

#pragma omp parallel 
    {
        int ThreadID = omp_get_thread_num();

        // Определяем границы блока для текущего потока
        int block_row = ThreadID / GridSize;
        int block_col = ThreadID % GridSize;

        int i_start = block_row * BlockSize + 1;
        int i_end = i_start + BlockSize;
        int j_start = block_col * BlockSize + 1;
        int j_end = j_start + BlockSize;

        // Вычисляем свой блок
        for (int i = i_start; i < i_end; ++i) {
            for (int j = j_start; j < j_end; ++j) {
                double sum = 0.0;
                for (int k = 1; k <= size; ++k) {
                    sum += matrixA[i][k] * matrixB[k][j];
                }
                result[i][j] = sum;
            }
        }
    }

    double end = omp_get_wtime();
    double duration = end - start;

    cout << "Результат умножения (блочное разбиение):\n";
    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            cout << fixed << setprecision(6) << result[i][j] << "\t";
        }
        cout << "\n";
    }
    cout << "Время выполнения: " << duration << " секунд\n\n";
}


void ProcessTerminate(double** matrixA, double** matrixB, double** result, int size) {
    FreeMatrix(matrixA, size);
    FreeMatrix(matrixB, size);
    FreeMatrix(result, size);

    matrixA = nullptr;
    matrixB = nullptr;
    result = nullptr;

    cout << "Память освобождена.\n";
}

int main() {
    setlocale(LC_ALL, "rus");
    double** matrixA = nullptr;
    double** matrixB = nullptr;
    double** result = nullptr;
    int size = 0;

    ProcessInit(matrixA, matrixB, result, size);
    SerialProduct(matrixA, matrixB, result, size);
    ParallelProduct_Row(matrixA, matrixB, result, size);
    ParallelProduct_Column(matrixA, matrixB, result, size);
    ParallelProduct_Block(matrixA, matrixB, result, size);
    ProcessTerminate(matrixA, matrixB, result, size);

    return 0;
}