//nvc++ -std=c++11 -mp=gpu -acc -gpu=nomanaged -Minfo -o manag managedvsexplicit.cu
#include <iostream>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <openacc.h>
#include <cuda_runtime.h>

using namespace std;

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

// Matrix multiplication using OpenACC with managed memory
void matrixMultiplyOpenACCManaged(int n, double *A, double *B, double *C, int num_devices)
{
#pragma omp parallel num_threads(num_devices)
    {
        int thread_id = omp_get_thread_num();
        int rows_per_device = (n + num_devices - 1) / num_devices; // Adjust rows per device to handle non-even distribution
        int start_row = thread_id * rows_per_device;
        int end_row = min((thread_id + 1) * rows_per_device, n);

        if (start_row < end_row) {
            acc_set_device_num(thread_id, acc_device_nvidia);

#pragma acc parallel loop collapse(2)
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

// Function to check if a pointer is managed
bool isManagedMemory(void *ptr)
{
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    return attributes.type == cudaMemoryTypeManaged;
}

int main2(const int a)
{
    const int n = 1 << 11; // Size of the matrices (reduced for testing)
    // Allocate managed memory arrays
    double *A = new double[n * n];
    double *B = new double[n * n];
    double *C_openmp = new double[n * n];
    double *C_openacc = new double[n * n];
    double *Am, *Bm, *C_opacc_managed;
    cudaMallocManaged(&Am, n * n * sizeof(double));
    cudaMallocManaged(&Bm, n * n * sizeof(double));
    cudaMallocManaged(&C_opacc_managed, n * n * sizeof(double));

    // Check if managed memory is enabled
    if (isManagedMemory(A))
    {
        cout << "Managed memory is enabled for A" << endl;
    }
    if (isManagedMemory(B))
    {
        cout << "Managed memory is enabled for B" << endl;
    }
    if (isManagedMemory(C_openmp))
    {
        cout << "Managed memory is enabled for C_openmp" << endl;
    }
    if (isManagedMemory(C_openacc))
    {
        cout << "Managed memory is enabled for C_openacc" << endl;
    }
    if (isManagedMemory(Am))
    {
        cout << "Managed memory is enabled for Am" << endl;
    }
    if (isManagedMemory(Bm))
    {
        cout << "Managed memory is enabled for Bm" << endl;
    }

    if (isManagedMemory(C_opacc_managed))
    {
        cout << "Managed memory is enabled for C_opacc_managed" << endl;
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = static_cast<double>((i * n + j) % 100);       // Modulo to keep values small
            B[i * n + j] = static_cast<double>(((i * n + j) + 1) % 100); // Modulo to keep values small
            Am[i * n + j] = static_cast<double>((i * n + j) % 100);       // Modulo to keep values small
            Bm[i * n + j] = static_cast<double>(((i * n + j) + 1) % 100); // Modulo to keep values small
            C_openmp[i * n + j] = 0.0;
            C_openacc[i * n + j] = 0.0;
            C_opacc_managed[i * n + j] = 0.0;
        }
    }

    // Print OpenMP device information
    int num_devices = a;
    // omp_get_num_devices();
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
    int acc_num_devices = a;
    // acc_get_num_devices(acc_device_nvidia);
    cout << "Number of OpenACC devices available: " << acc_num_devices << endl;

    if (acc_num_devices > 0)
    {
        // Time taken for matrix multiplication using OpenACC
        auto start_openacc = chrono::high_resolution_clock::now();
        matrixMultiplyOpenACC(n, A, B, C_openacc, acc_num_devices);
        auto end_openacc = chrono::high_resolution_clock::now();
        chrono::duration<double> duration_openacc = end_openacc - start_openacc;
        cout << "Time taken for matrix multiplication using OpenACC: " << duration_openacc.count() << " seconds" << endl;

        // Time taken for matrix multiplication using OpenACC with managed memory
        auto start_opacc_managed = chrono::high_resolution_clock::now();
        matrixMultiplyOpenACCManaged(n, Am, Bm, C_opacc_managed, acc_num_devices);
        auto end_opacc_managed = chrono::high_resolution_clock::now();
        chrono::duration<double> duration_opacc_managed = end_opacc_managed - start_opacc_managed;
        cout << "Time taken for matrix multiplication using OpenACC (Managed): " << duration_opacc_managed.count() << " seconds" << endl;
    }
    else
    {
        cout << "No OpenACC devices available" << endl;
    }

    // Compare the results of all methods
    bool result_correct_openmp = compareMatrices(n, C_openmp, C_openacc);
    bool result_correct_opacc_managed = compareMatrices(n, C_openacc, C_opacc_managed);

    if (result_correct_openmp && result_correct_opacc_managed)
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
    cout << "C_opacc_managed[0] = " << C_opacc_managed[0] << endl;

    // Cleanup managed memory
    delete[] A;
    delete[] B;
    delete[] C_openmp;
    delete[] C_openacc;
    cudaFree(Am);
    cudaFree(Bm);
    cudaFree(C_opacc_managed);

    return 0;
}

int main()
{
    for(int i = 1; i< 5; i++)
    {
        main2(i);
    }
}
