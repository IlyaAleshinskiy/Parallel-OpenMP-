#include <iostream>
#include <iomanip>
#include <omp.h>

using namespace std;

void ProcessInit(double**& matrix, double*& vector, double*& result, int& size) {
    cout << "Введите размер квадратной матрицы (n): "; cin >> size;
    matrix = new double* [size + 1];
    for (int i = 0; i <= size; ++i) {
        matrix[i] = new double[size + 1]();
    }

    vector = new double[size + 1]();
    result = new double[size + 1]();

    cout << "Введите элементы матрицы A (" << size << "x" << size << "):\n";
    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            cout << "A[" << i << "][" << j << "] = ";
            cin >> matrix[i][j];
        }
    }

    cout << "Введите элементы вектора b (" << size << " элементов):\n";
    for (int j = 1; j <= size; ++j) {
        cout << "b[" << j << "] = ";
        cin >> vector[j];
    }
    cout << "Инициализация завершена.\n\n";
}

void SerialProduct(double** matrix, double* vector, double* result, int size) {
    double start = omp_get_wtime();

    for (int i = 1; i <= size; ++i) {
        result[i] = 0.0;
        for (int j = 1; j <= size; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    double end = omp_get_wtime();
    double duration = end - start;

    cout << "Результат последовательного умножения (вектор c):\n";
    for (int i = 1; i <= size; ++i) {
        cout << "c[" << i << "] = " << fixed << setprecision(8) << result[i] << "\n";
    }
    cout << "Время выполнения (последовательно): " << duration << " нс\n\n";
}

void ParallelProduct_Row(double** matrix, double* vector, double* result, int size) {
    double start = omp_get_wtime();
    int j;
#pragma omp parallel for private(j)
    for (int i = 1; i <= size; ++i) {
        double sum = 0.0;
        for (j = 1; j <= size; ++j) {
            sum += matrix[i][j] * vector[j];
        }
        result[i] = sum;
    }

    double end = omp_get_wtime();
    double duration = end - start;

    cout << "Результат параллельного умножения (по строкам):\n";
    for (int i = 1; i <= size; ++i) {
        cout << "c[" << i << "] = " << fixed << setprecision(8) << result[i] << "\n";
    }
    cout << "Время выполнения (параллельно по строкам): " << duration << " нс\n\n";
}
void ParallelProduct_Column(double** matrix, double* vector, double* result, int size) {
    double start = omp_get_wtime();

    for (int i = 1; i <= size; ++i) {
        result[i] = 0.0;
    }

#pragma omp parallel for
    for (int j = 1; j <= size; ++j) {
        double column = vector[j];
        // Каждый поток обрабатывает свой столбец j
        for (int i = 1; i <= size; ++i) {
            // Критическая секция или атомарная операция нужна,
            // потому что разные j могут обновлять один и тот же result[i]
#pragma omp atomic
            result[i] += matrix[i][j] * column;
        }
    }

    double end = omp_get_wtime();
    double duration = end - start;

    cout << "Результат умножения (по столбцам):\n";
    for (int i = 1; i <= size; ++i) {
        cout << "c[" << i << "] = " << fixed << setprecision(6) << result[i] << "\n";
    }
    cout << "Время выполнения (паралелльно по столбцам): " << duration << " нс\n\n";
}
void ParallelProduct_Block(double** matrix, double* vector, double* result, int size) {
    double start = omp_get_wtime();

    // Определяем количество потоков (должно быть квадратом!)
    const int GridThreadsNum = 4; 
    const int GridSize = sqrt(GridThreadsNum);
    const int BlockSize = size / GridSize;

    // Проверка: Size должно делиться на GridSize
    if (size % GridSize != 0) {
        cerr << "Ошибка: размер матрицы должен делиться на " << GridSize << " без остатка!\n";
        return;
    }

    // Обнуляем результат
    for (int i = 1; i <= size; ++i) {
        result[i] = 0.0;
    }

    omp_set_num_threads(GridThreadsNum);

#pragma omp parallel
    {
        int ThreadID = omp_get_thread_num();
        // Локальный буфер для вклада этого потока
        double* pThreadResult = new double[size + 1](); // индексы 0..Size, используем 1..Size

        int i_start = (ThreadID / GridSize) * BlockSize + 1; // +1 для индексации с 1
        int j_start = (ThreadID % GridSize) * BlockSize + 1;

        // Обрабатываем блок: строки [i_start, i_start + BlockSize), столбцы [j_start, j_start + BlockSize)
        for (int i = 0; i < BlockSize; ++i) {
            int row = i_start + i;
            for (int j = 0; j < BlockSize; ++j) {
                int col = j_start + j;
                pThreadResult[row] += matrix[row][col] * vector[col];
            }
        }

        // Критическая секция: добавляем вклад в общий результат
#pragma omp critical
        {
            for (int i = 1; i <= size; ++i) {
                result[i] += pThreadResult[i];
            }
        }

        delete[] pThreadResult;
    }

    double end = omp_get_wtime();
    double duration = end - start;

    cout << "Результат умножения (блочное разбиение):\n";
    for (int i = 1; i <= size; ++i) {
        cout << "c[" << i << "] = " << fixed << setprecision(6) << result[i] << "\n";
    }
    cout << "Время выполнения (блочное разбиение): " << duration << " нс\n\n";
}

void ProcessTerminate(double** matrix, double* vector, double* result, int size) {
    for (int i = 0; i <= size; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;

    delete[] vector;
    delete[] result;

    matrix = nullptr;
    vector = nullptr;
    result = nullptr;
    size = 0;

    cout << "Память освобождена.\n";
}

int main() {
    setlocale(LC_ALL, "rus");
    double** matrix = nullptr;
    double* vector = nullptr;
    double* result = nullptr;
    int size = 0;
    

    ProcessInit(matrix, vector, result, size);
    SerialProduct(matrix, vector, result, size);
    ParallelProduct_Row(matrix, vector, result, size);
    ParallelProduct_Column(matrix, vector, result, size);
    ParallelProduct_Block(matrix, vector, result, size);
    ProcessTerminate(matrix, vector, result, size);
    return 0;
}
