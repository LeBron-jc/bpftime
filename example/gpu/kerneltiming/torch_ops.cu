// 算子→kernel 多层映射演示
// 模拟真实场景：一个算子可能触发多个 kernel，不同算子可能共用 kernel

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

__constant__ int d_N;

// ── 三个 GPU kernel ──
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

__global__ void relu(float *C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < d_N && C[idx] < 0) C[idx] = 0;
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
	cudaMemset(d_C, 0, bytes);
	delete[] tmp;
}

// ═══════════════════════════════════════════════════════
// 三个算子，模拟真实 PyTorch 内部结构
// ═══════════════════════════════════════════════════════

namespace at { namespace native {

// matmul: 一个算子触发 2 个 kernel（加 + 乘加）
__attribute__((noinline))
void matmul_out(const float *A, const float *B, float *C)
{
	vectorAdd<<<4, 256>>>(A, B, C);
	cudaDeviceSynchronize();
	multiplyAdd<<<4, 256>>>(A, B, C);
	cudaDeviceSynchronize();
}

// relu: 一个算子只触发 1 个 kernel
__attribute__((noinline))
void relu_out(float *C)
{
	relu<<<4, 256>>>(C);
	cudaDeviceSynchronize();
}

// addmm: 另一个算子也触发多个 kernel（和 matmul 共享 multiplyAdd）
__attribute__((noinline))
void addmm_out(const float *A, const float *B, float *C)
{
	multiplyAdd<<<4, 256>>>(A, B, C);
	cudaDeviceSynchronize();
	relu<<<4, 256>>>(C);
	cudaDeviceSynchronize();
}

} } // namespace at::native

int main()
{
	init();
	while (true) {
		at::native::matmul_out(d_A, d_B, d_C);
		at::native::relu_out(d_C);
		at::native::addmm_out(d_A, d_B, d_C);
		sleep(1);
	}
	return 0;
}
