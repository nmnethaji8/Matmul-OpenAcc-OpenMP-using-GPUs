// nvc++ -std=c++11 -mp=gpu -acc -Minfo -o matrix_multiply matrix_multiply.cpp
#include <iostream>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <openacc.h>

using namespace std;

// Matrix multiplication using OpenMP target offloading
void matrixMultiplyOpenMP(int n, double* A, double* B, double* C) {
    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: A[0:n*n], B[0:n*n]) map(from: C[0:n*n])
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Matrix multiplication using OpenACC distributed among multiple GPUs
void matrixMultiplyOpenACC(int n, double* A, double* B, double* C, int num_devices) {
    #pragma omp parallel num_threads(num_devices)
    {
        int thread_id = omp_get_thread_num();
        int rows_per_device = n / num_devices;
        int start_row = thread_id * rows_per_device;
        int end_row = (thread_id == num_devices - 1) ? n : start_row + rows_per_device;

        acc_set_device_num(thread_id, acc_device_nvidia);

        #pragma acc parallel loop collapse(2) copyin(A[0:n*n], B[0:n*n]) copyout(C[start_row*n:(end_row-start_row)*n])
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
    }
}

// Function to compare results of two matrices
bool compareMatrices(int n, double* C1, double* C2) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(C1[i] - C2[i]) > 1e-10) {
            return false;
        }
    }
    return true;
}

int main() {
    const int n = 10240; // Size of the matrices
    double *A = new double[n * n];
    double *B = new double[n * n];
    double *C_openmp = new double[n * n];
    double *C_openacc = new double[n * n];

    // Initialize matrices A and B with some values
    for (int i = 0; i < n * n; i++) {
        A[i] = static_cast<double>(i % 100); // Modulo to keep values small
        B[i] = static_cast<double>((i + 1) % 100); // Modulo to keep values small
        C_openmp[i] = 0.0;
        C_openacc[i] = 0.0;
    }

    // Print OpenMP device information
    int num_devices = omp_get_num_devices();
    cout << "Number of OpenMP devices available: " << num_devices << endl;

    // Perform matrix multiplication using OpenMP and time it
    auto start_openmp = chrono::high_resolution_clock::now();
    matrixMultiplyOpenMP(n, A, B, C_openmp);
    auto end_openmp = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_openmp = end_openmp - start_openmp;
    cout << "Time taken for matrix multiplication using OpenMP: " << duration_openmp.count() << " seconds" << endl;

    // Perform matrix multiplication using OpenACC on multiple GPUs
    int acc_num_devices = acc_get_num_devices(acc_device_nvidia);
    cout << "Number of OpenACC devices available: " << acc_num_devices << endl;

    if (acc_num_devices > 0) {
        // Time taken for matrix multiplication using OpenACC
        auto start_openacc = chrono::high_resolution_clock::now();
        matrixMultiplyOpenACC(n, A, B, C_openacc, acc_num_devices);
        auto end_openacc = chrono::high_resolution_clock::now();
        chrono::duration<double> duration_openacc = end_openacc - start_openacc;
        cout << "Time taken for matrix multiplication using OpenACC: " << duration_openacc.count() << " seconds" << endl;
    } else {
        cout << "No OpenACC devices available" << endl;
        return 1; // Exit if no devices available
    }

    // Compare the results of both methods
    bool result_correct = compareMatrices(n, C_openmp, C_openacc);
    if (result_correct) {
        cout << "The results are correct!" << endl;
    } else {
        cout << "The results are incorrect!" << endl;
    }

    // Optionally, print part of the result
    cout << "C_openmp[0][0] = " << C_openmp[0] << endl;
    cout << "C_openacc[0][0] = " << C_openacc[0] << endl;

    // Cleanup dynamically allocated memory
    delete[] A;
    delete[] B;
    delete[] C_openmp;
    delete[] C_openacc;

    return 0;
}
