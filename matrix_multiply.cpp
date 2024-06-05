// nvc++ -std=c++11 -mp=gpu -acc -Minfo -L/apps/ACC/NVIDIA-HPC-SDK/24.3/Linux_x86_64/24.3/math_libs/lib64 -L/apps/ACC/NVIDIA-HPC-SDK/24.3/Linux_x86_64/24.3/cuda/12.3/targets/x86_64-linux/lib/ -lcublas -lcudart -o matrix_multiply matrix_multiply.cpp
#include <iostream>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <openacc.h>
#include <cublas_v2.h>
#include <cublasXt.h>

using namespace std;

#define CUDA_CHECK(call)                                                            \
    {                                                                               \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                     \
        {                                                                           \
            cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl;                                \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }

#define CUBLAS_CHECK(call)                                                            \
    {                                                                                 \
        cublasStatus_t err = call;                                                    \
        if (err != CUBLAS_STATUS_SUCCESS)                                             \
        {                                                                             \
            cerr << "cuBLAS error in " << __FILE__ << " at line " << __LINE__ << ": " \
                 << err << endl;                                                      \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    }

// Matrix multiplication using OpenMP target offloading
void matrixMultiplyOpenMP(int n, double *A, double *B, double *C, int num_devices)
{
#pragma omp parallel num_threads(num_devices)
    {
        int thread_id = omp_get_thread_num();
        omp_set_default_device(thread_id); // Set the device for this thread

        int rows_per_device = (n + num_devices - 1) / num_devices; // Adjust rows per device to handle non-even distribution
        int start_row = thread_id * rows_per_device;
        int end_row = min((thread_id + 1) * rows_per_device, n);

        if (start_row < end_row) {
#pragma omp target teams distribute parallel for collapse(2) map(to: A[start_row * n : (end_row - start_row) * n], B[0 : n * n]) map(from: C[start_row * n : (end_row - start_row) * n])
            for (int i = start_row; i < end_row; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += A[i * n + k] * B[k * n + j];
                    }
                    C[i * n + j] = sum;
                }
            }
        }
    }
}

// Matrix multiplication using OpenACC distributed among multiple GPUs
void matrixMultiplyOpenACC(int n, double *A, double *B, double *C, int num_devices)
{
#pragma omp parallel num_threads(num_devices)
    {
        int thread_id = omp_get_thread_num();
        int rows_per_device = (n + num_devices - 1) / num_devices; // Adjust rows per device to handle non-even distribution
        int start_row = thread_id * rows_per_device;
        int end_row = min((thread_id + 1) * rows_per_device, n);

        if (start_row < end_row) {
            acc_set_device_num(thread_id, acc_device_nvidia);

#pragma acc parallel loop collapse(2) copyin(A[start_row * n : (end_row - start_row) * n], B[0 : n * n]) copyout(C[start_row * n : (end_row - start_row) * n])
            for (int i = start_row; i < end_row; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += A[i * n + k] * B[k * n + j];
                    }
                    C[i * n + j] = sum;
                }
            }
        }
    }
}

void matrixMultiplyCuBLAS(int n, double *A, double *B, double *C, const int num_gpus)
{
    // Create cuBLAS Xt handle
    cublasXtHandle_t handle;
    CUBLAS_CHECK(cublasXtCreate(&handle));

    // Set the number of devices
    int devices[num_gpus];
    for (int i = 0; i < num_gpus; ++i)
    {
        devices[i] = i;
    }

    CUBLAS_CHECK(cublasXtDeviceSelect(handle, num_gpus, devices));

    // Perform matrix multiplication on each GPU, considering row-major order
    const double alpha = 1.0;
    const double beta = 0.0;
    CUBLAS_CHECK(cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, n, A, n, &beta, C, n));

    // Destroy cuBLAS Xt handle
    CUBLAS_CHECK(cublasXtDestroy(handle));
}

// Function to compare results of two matrices
bool compareMatrices(int n, double *C1, double *C2)
{
    double error = 0.0;
#pragma omp parallel for reduction(+: error)
    for (int i = 0; i < n * n; i++)
    {
        error += fabs(C1[i] - C2[i]);
    }

    return error <= 1e-10;
}

int main()
{
    const int n = 1 << 5; // Size of the matrices (reduced for testing)

    // Allocate 1D arrays
    double *A = new double[n * n];
    double *B = new double[n * n];
    double *C_openmp = new double[n * n];
    double *C_openacc = new double[n * n];
    double *C_cublas = new double[n * n];

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = static_cast<double>((i * n + j) % 100);       // Modulo to keep values small
            B[i * n + j] = static_cast<double>(((i * n + j) + 1) % 100); // Modulo to keep values small
            C_openmp[i * n + j] = 0.0;
            C_openacc[i * n + j] = 0.0;
            C_cublas[i * n + j] = 0.0;
        }
    }

    // Print OpenMP device information
    int num_devices = omp_get_num_devices();
    cout << "Number of OpenMP devices available: " << num_devices << endl;

    if (num_devices > 0)
    {
        // Perform matrix multiplication using OpenMP and time it
        auto start_openmp = chrono::high_resolution_clock::now();
        matrixMultiplyOpenMP(n, A, B, C_openmp, num_devices);
        auto end_openmp = chrono::high_resolution_clock::now();
        chrono::duration<double> duration_openmp = end_openmp - start_openmp;
        cout << "Time taken for matrix multiplication using OpenMP: " << duration_openmp.count() << " seconds" << endl;
    }
    else
    {
        cout << "No OpenMP devices available" << endl;
    }

    // Perform matrix multiplication using OpenACC on multiple GPUs
    int acc_num_devices = acc_get_num_devices(acc_device_nvidia);
    cout << "Number of OpenACC devices available: " << acc_num_devices << endl;

    if (acc_num_devices > 0)
    {
        // Time taken for matrix multiplication using OpenACC
        auto start_openacc = chrono::high_resolution_clock::now();
        matrixMultiplyOpenACC(n, A, B, C_openacc, acc_num_devices);
        auto end_openacc = chrono::high_resolution_clock::now();
        chrono::duration<double> duration_openacc = end_openacc - start_openacc;
        cout << "Time taken for matrix multiplication using OpenACC: " << duration_openacc.count() << " seconds" << endl;

        // Perform matrix multiplication using cuBLAS and time it
        auto start_cublas = chrono::high_resolution_clock::now();
        matrixMultiplyCuBLAS(n, A, B, C_cublas, acc_num_devices);
        auto end_cublas = chrono::high_resolution_clock::now();
        chrono::duration<double> duration_cublas = end_cublas - start_cublas;
        cout << "Time taken for matrix multiplication using cuBLAS: " << duration_cublas.count() << " seconds" << endl;
    }
    else
    {
        cout << "No OpenACC/CUDA devices available" << endl;
    }

    // Compare the results of both methods
    bool result_correct_openmp = compareMatrices(n, C_openmp, C_cublas);
    bool result_correct_openacc = compareMatrices(n, C_openacc, C_cublas);

    if (result_correct_openmp && result_correct_openacc)
    {
        cout << "The results are correct!" << endl;
    }
    else
    {
        cout << "The results are incorrect!" << endl;
    }

    // Optionally, print part of the result
    cout << "C_openmp[0] = " << C_openmp[0] << endl;
    cout << "C_openacc[0] = " << C_openacc[0] << endl;
    cout << "C_cublas[0] = " << C_cublas[0] << endl;

    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         cout << C_openmp[i * n + j] << "\t";
    //     }
    //     cout << "\n";
    // }

    // cout << "****************************\n";
    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         cout << C_cublas[i * n + j] << "\t";
    //     }
    //     cout << "\n";
    // }
    // Cleanup dynamically allocated memory
    delete[] A;
    delete[] B;
    delete[] C_openmp;
    delete[] C_openacc;
    delete[] C_cublas;

    return 0;
}
