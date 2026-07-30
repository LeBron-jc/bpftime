// Simulates a PyTorch operator calling a CUDA kernel.
// The function name contains "at::native::" so that bpftime's
// resolve_pytorch_caller_from_backtrace() picks it up as operator name.

#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h>

__constant__ int d_N;

__global__ void vectorAdd(const float *A, const float *B, float *C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < d_N) C[idx] = A[idx] + B[idx];
}

__global__ void multiplyAdd(const float *A, const float *B, float *C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < d_N) C[idx] = A[idx] * B[idx] + 1.0f;
}

static float *d_A, *d_B, *d_C;
static const int N = 1 << 20;

static void init()
{
	cudaMemcpyToSymbol(d_N, &N, sizeof(N));
	size_t bytes = N * sizeof(float);
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);
	float *tmp = new float[N];
	for (int i = 0; i < N; i++) tmp[i] = (float)i;
	cudaMemcpy(d_A, tmp, bytes, cudaMemcpyHostToDevice);
	for (int i = 0; i < N; i++) tmp[i] = (float)(2 * i);
	cudaMemcpy(d_B, tmp, bytes, cudaMemcpyHostToDevice);
	delete[] tmp;
}

namespace at { namespace native {

__attribute__((noinline))
void matmul_out()
{
	vectorAdd<<<4, 256>>>(d_A, d_B, d_C);
	cudaDeviceSynchronize();
}

__attribute__((noinline))
void relu_out()
{
	multiplyAdd<<<4, 256>>>(d_A, d_B, d_C);
	cudaDeviceSynchronize();
}

} }

int main()
{
	init();
	while (true) {
		at::native::matmul_out();
		at::native::relu_out();
		sleep(1);
	}
	return 0;
}
