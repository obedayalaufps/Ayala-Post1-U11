#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// 1. Kernel Naïve: Acceso directo e individual a memoria global
__global__ void matMulNaive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 2. Kernel Tiled: Uso colaborativo y eficiente de Shared Memory
__global__ void matMulTiled(const float *A, const float *B, float *C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (t * TILE_SIZE + threadIdx.y) < N) {
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads(); // Sincronización obligatoria: Esperar la carga del tile

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads(); // Esperar antes de sobreescribir el tile en el siguiente ciclo
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void runBenchmark(int N) {
    size_t bytes = N * N * sizeof(float);
    printf("\n--- Evaluando Matriz de dimensiones: %d x %d ---\n", N, N);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_naive = (float*)malloc(bytes);
    float *h_C_tiled = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Ejecución y cronometraje de la versión Naïve
    cudaEventRecord(start);
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);
    cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost);

    // Ejecución y cronometraje de la versión Tiled
    cudaEventRecord(start);
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_tiled = 0;
    cudaEventElapsedTime(&ms_tiled, start, stop);
    cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost);

    // Validación de consistencia matemática entre algoritmos
    int correct = 1;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C_naive[i] - h_C_tiled[i]) > 1e-3) {
            correct = 0;
            break;
        }
    }

    printf("Resultado del analisis: %s\n", correct ? "CORRECTO (Error < 1e-3)" : "INCORRECTO");
    printf("Tiempo Kernel Naive : %.2f ms\n", ms_naive);
    printf("Tiempo Kernel Tiled : %.2f ms\n", ms_tiled);
    printf("Factor de aceleracion (Speedup): %.2f x\n", ms_naive / ms_tiled);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled);
}

int main() {
    runBenchmark(512);
    runBenchmark(1024);
    return 0;
}
