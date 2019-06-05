//
// Created by wei on 2/17/18.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cuda_runtime.h>
//#include "cublas_v2.h"
#include <cublas_v2.h>

#include "common/sanity_check.h"

#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
static __inline__ void modify (
        cublasHandle_t handle,
        float *m, int ldm, int n, int p, int q, float alpha, float beta
){
    cublasSscal (handle, n-p, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}

void test_dot_product(cublasHandle_t handle) {
    using namespace surfelwarp;

    std::vector<float> vec_0, vec_1;
    vec_0.resize(1000);
    vec_1.resize(1000);

    fillRandomVector(vec_0);
    fillRandomVector(vec_1);
    float dot_host = 0.0f;
    for (int i = 0; i < vec_0.size(); ++i) {
        dot_host += vec_0[i] * vec_1[i];
    }

    //Upload to device
    DeviceArray<float> d_vec_0, d_vec_1, d_result;
    d_vec_0.upload(vec_0);
    d_vec_1.upload(vec_1);
    d_result.create(10);

    //Set the pointer mode
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSdot(handle, d_vec_0.size(), d_vec_0.ptr(), 1, d_vec_1.ptr(), 1, d_result.ptr());
    cublasSdot(handle, d_vec_0.size(), d_vec_0.ptr(), 1, d_vec_1.ptr(), 1, d_result.ptr());
    //Download it
    std::vector<float> d_result_h;
    d_result.download(d_result_h);
    float diff = d_result_h[0] - dot_host;
    assert(std::abs(diff) < 1e-4);
}


int main (void)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    float* devPtrA;
    float* a = 0;
    a = (float *)malloc (M * N * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            a[IDX2C(i,j,M)] = (float)(i * M + j + 1);
        }
    }

    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);

    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaFree (devPtrA);
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", a[IDX2C(i,j,M)]);
        }
        printf ("\n");
    }

    free(a);

    //Test of dot product
    test_dot_product(handle);
    cublasDestroy(handle);
    return EXIT_SUCCESS;
}

