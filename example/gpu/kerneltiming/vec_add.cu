#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <vector>

__constant__ int d_N;

// Kernel with intentional block-level execution time divergence:
// even blocks do extra work, odd blocks return early.
// This produces clear per-thread duration differences in bpftime output.
__global__ void vectorAdd(const float *A, const float *B, float *C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < d_N) {
		C[idx] = A[idx] + B[idx];
	}

	// Even-numbered blocks do extra computation → longer execution time
	if (blockIdx.x % 2 == 0) {
		float acc = 0.0f;
		for (int i = 0; i < 5000; i++) {
			acc = acc * 0.999f + 0.001f;
		}
		if (idx < d_N && threadIdx.x == 0) {
			C[idx] = acc;
		}
	}
}

int main()
{
	const int h_N = 1 << 20; // 1M elements
	cudaMemcpyToSymbol(d_N, &h_N, sizeof(h_N));

	size_t bytes = h_N * sizeof(float);

	std::vector<float> h_A(h_N), h_B(h_N), h_C(h_N);

	for (int i = 0; i < h_N; ++i) {
		h_A[i] = float(i);
		h_B[i] = float(2 * i);
	}

	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

	// 4 blocks × 256 threads = 1024 threads total
	// Blocks 0,2 will be slow (extra loop), blocks 1,3 will be fast
	while (true) {
		cudaMemset(d_C, 0, bytes);

		vectorAdd<<<4, 256>>>(d_A, d_B, d_C);
		cudaDeviceSynchronize();

		cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

		std::cout << "C[0] = " << h_C[0] << " (expected 0)" << std::endl;
		std::cout << "C[1] = " << h_C[1] << " (expected 3)" << std::endl;

		sleep(1);
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
