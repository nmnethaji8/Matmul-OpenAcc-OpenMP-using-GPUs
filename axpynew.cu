#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <omp.h>
#include <openacc.h>

using namespace std;

// CUDA AXPY Kernel
__global__ void axpy_kernel(int n, double alpha, double *X, double *Y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        Y[i] = alpha * X[i] + Y[i];
    }
}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")" << endl;
        exit(EXIT_FAILURE);
    }
}

// CUDA AXPY Multi-GPU Function
void axpy_cuda_multi_gpu(int n, double alpha, double *X, double *Y, int num_devices)
{
    int chunk_size = (n + num_devices - 1) / num_devices;
    double **d_X = new double *[num_devices];
    double **d_Y = new double *[num_devices];
    cudaStream_t *streams = new cudaStream_t[num_devices];

#pragma omp parallel for num_threads(num_devices)
    for (int i = 0; i < num_devices; ++i)
    {
        checkCudaError(cudaSetDevice(i), "Failed to set device");
        checkCudaError(cudaStreamCreate(&streams[i]), "Failed to create stream");

        checkCudaError(cudaMalloc(&d_X[i], chunk_size * sizeof(double)), "Failed to allocate device memory for X");
        checkCudaError(cudaMalloc(&d_Y[i], chunk_size * sizeof(double)), "Failed to allocate device memory for Y");

        int offset = i * chunk_size;
        int current_chunk_size = min(chunk_size, n - offset);

        checkCudaError(cudaMemcpyAsync(d_X[i], X + offset, current_chunk_size * sizeof(double), cudaMemcpyHostToDevice, streams[i]), "Failed to copy X to device asynchronously");
        checkCudaError(cudaMemcpyAsync(d_Y[i], Y + offset, current_chunk_size * sizeof(double), cudaMemcpyHostToDevice, streams[i]), "Failed to copy Y to device asynchronously");

        int blockSize = 256;
        int numBlocks = (current_chunk_size + blockSize - 1) / blockSize;

        axpy_kernel<<<numBlocks, blockSize, 0, streams[i]>>>(current_chunk_size, alpha, d_X[i], d_Y[i]);
        checkCudaError(cudaGetLastError(), "Kernel launch failed");

        checkCudaError(cudaMemcpyAsync(Y + offset, d_Y[i], current_chunk_size * sizeof(double), cudaMemcpyDeviceToHost, streams[i]), "Failed to copy Y back to host asynchronously");
        checkCudaError(cudaStreamSynchronize(streams[i]), "Stream synchronization failed");

        checkCudaError(cudaFree(d_X[i]), "Failed to free device memory for X");
        checkCudaError(cudaFree(d_Y[i]), "Failed to free device memory for Y");
        checkCudaError(cudaStreamDestroy(streams[i]), "Failed to destroy stream");
    }

    delete[] d_X;
    delete[] d_Y;
    delete[] streams;
}

// OpenMP AXPY Multi-GPU Function
void axpy_openmp_multi_gpu(int n, double alpha, double *X, double *Y, int num_devices)
{
#pragma omp parallel num_threads(num_devices)
    {
        int tid = omp_get_thread_num();
        omp_set_default_device(tid);
        int chunk_size = n / num_devices;
        int start = tid * chunk_size;
        int end = (tid == num_devices - 1) ? n : start + chunk_size;

#pragma omp target teams distribute parallel for map(to : X[start : chunk_size]) map(tofrom : Y[start : chunk_size])
        for (int i = start; i < end; i++)
        {
            Y[i] = alpha * X[i] + Y[i];
        }
    }
}

// OpenACC AXPY Multi-GPU Function
void axpy_openacc_multi_gpu(int n, double alpha, double *X, double *Y, int num_devices)
{
#pragma omp parallel num_threads(num_devices)
    {
        int tid = omp_get_thread_num();
        omp_set_default_device(tid);
        int chunk_size = n / num_devices;
        int start = tid * chunk_size;
        int end = (tid == num_devices - 1) ? n : start + chunk_size;

#pragma acc parallel loop copyin(X[start : chunk_size]) copy(Y[start : chunk_size])
        for (int i = start; i < end; i++)
        {
            Y[i] = alpha * X[i] + Y[i];
        }
    }
}

int main()
{
    const int n = 1 << 10; // Size of vectors
    const double alpha = 2.0;

    // Allocate memory for vectors
    double *X = new double[n];
    double *Y = new double[n];
    double *Y_openmp = new double[n];
    double *Y_openacc = new double[n];

    // Initialize vectors
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        X[i] = i;
        Y[i] = 2 * i;
        Y_openmp[i] = 2 * i;
        Y_openacc[i] = 2 * i;
    }

    // Get the number of devices
    int num_cuda_devices;
    checkCudaError(cudaGetDeviceCount(&num_cuda_devices), "Failed to get CUDA device count");
    int num_openmp_devices = omp_get_num_devices();

    // CUDA AXPY
    auto start = chrono::high_resolution_clock::now();
    axpy_cuda_multi_gpu(n, alpha, X, Y, num_cuda_devices);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_cuda = end - start;
    cout << "Execution time of AXPY operation using CUDA on multiple GPUs: " << duration_cuda.count() << " seconds" << endl;

    // OpenMP AXPY
    start = chrono::high_resolution_clock::now();
    axpy_openmp_multi_gpu(n, alpha, X, Y_openmp, num_openmp_devices);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_openmp = end - start;
    cout << "Execution time of AXPY operation using OpenMP on multiple GPUs: " << duration_openmp.count() << " seconds" << endl;

    // OpenACC AXPY
    start = chrono::high_resolution_clock::now();
    axpy_openacc_multi_gpu(n, alpha, X, Y_openacc, num_openmp_devices);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_openacc = end - start;
    cout << "Execution time of AXPY operation using OpenACC on multiple GPUs: " << duration_openacc.count() << " seconds" << endl;

    // Validate the result
    bool result_correct = true;
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        if (Y[i] != (alpha * X[i] + 2 * i) || Y_openmp[i] != (alpha * X[i] + 2 * i) || Y_openacc[i] != (alpha * X[i] + 2 * i))
        {
            result_correct = false;
            cout << "Error at index " << i << ": CUDA " << Y[i] << ", OpenMP " << Y_openmp[i] << ", OpenACC " << Y_openacc[i] << endl;
            break;
        }
    }

    if (result_correct)
    {
        cout << "AXPY operation results are correct!" << endl;
    }
    else
    {
        cout << "AXPY operation results are incorrect!" << endl;
    }

    // Cleanup
    delete[] X;
    delete[] Y;
    delete[] Y_openmp;
    delete[] Y_openacc;

    return 0;
}
