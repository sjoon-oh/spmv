// Week 3
// cuSPARSE Example.
// acoustikue@yonsei.ac.kr
// written by SukJoon Oh

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cusparse.h>	// cusparseSpMV

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mmio.h"


#define CASE4
//#define CUSPARSE
#define CUSPARSE_CSR
// #define KERNEL_CSR_SCALAR
#define KERNEL_CSR_VECTOR

#ifdef CASE1
#define M	10
#define N	10
#define NZ	20
#define MTX_FILE	"10_10_sample_mat.mtx"
#endif
#ifdef CASE2
#define M	1024
#define N	1024
#define NZ	209715
#define MTX_FILE	"1024_1024_sample_mat.mtx"
#endif
#ifdef CASE3
#define M	2048
#define N	2048
#define NZ	838861
#define MTX_FILE	"2048_2048_sample_mat.mtx"
#endif
#ifdef CASE4
#define M	4096
#define N	4096
#define NZ	3355443
#define MTX_FILE	"4096_4096_sample_mat.mtx"
#endif

#define CUDA_ERR(func)                                                         \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CUSPARSE_ERR(func)                                                     \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


// Author: SukJoon Oh
// acoustikue@yonsei.ac.kr
// Reads MM file.
void read_matrix(int* argJR, int* argJC, float* argAA) {

	int m = M;
	int n = N;
	int nz = NZ;

	FILE* MTX;
	MTX = fopen(MTX_FILE, "r");
	 
	MM_typecode matrix_code;

	// Read banner, type, etc essential infos
	// Verification steps are ignored.
	if (mm_read_banner(MTX, &matrix_code) != 0) exit(1);
	mm_read_mtx_crd_size(MTX, &m, &n, &nz); // Over max 1025

	printf("Market Market type: [%s]\n", mm_typecode_to_str(matrix_code));

	// COO format
	for (register int i = 0; i < NZ; i++)
		fscanf_s(MTX, "%d %d %f\n", &argJR[i], &argJC[i], &argAA[i]);

	fclose(MTX);
}


// 
// CSR scalar kernel function
__global__ void ker_csr_spmv_scalar(
	const int* argJR, const int* argJC, const float* argAA,
	const float* arg_x, float* arg_y) {

	int idx		= blockDim.x * blockIdx.x + threadIdx.x;
	float sum	= 0;

	for (int i = argJR[idx] - 1; i < argJR[idx + 1] - 1; i++)
		sum		+= (argAA[i] * arg_x[argJC[i] - 1]);

	arg_y[idx]	+= sum;
};




//
// CSR vector kernel function
__global__ void ker_csr_spmv_vector(
	const int* argJR, const int* argJC, const float* argAA,
	const float* arg_x, float* arg_y) {

	// Thread : 32 * M

	int tid		= blockDim.x * blockIdx.x + threadIdx.x;
	int wid		= tid / 32;
	int lidx	= tid & 31;
	float sum	= 0;

	for (int i = argJR[wid] - 1 + lidx; i < argJR[wid + 1] - 1; i += 32)
		sum += argAA[i] * arg_x[argJC[i] - 1];

	for (int i = 16; i > 0; i /= 2)
		sum += __shfl_down_sync(0xFFFFFFFF, sum, i);

	if (lidx == 0) arg_y[wid] = sum;
};




// ---- main() ----
// Entry
int main()
{
	//
	// ---- Step 1. Load info ----
	int* host_JR	= (int*)malloc(NZ * sizeof(int));
	int* host_JC	= (int*)malloc(NZ * sizeof(int));
	float* host_AA	= (float*)malloc(NZ * sizeof(float));
	int* host_P		= (int*)malloc(NZ * sizeof(int));

	read_matrix(host_JR, host_JC, host_AA); // prepare elements

	//
	// ---- Step 2. Handle create, bind a stream ---- 
	int* device_JR			= NULL;
	int* device_JC			= NULL;
	float* device_AA		= NULL;
	float* device_AA_sorted	= NULL;
	int* device_P			= NULL;

	void* buffer			= NULL;
	size_t buffer_size		= 0;

	cusparseHandle_t handle = NULL;
	cudaStream_t stream		= NULL;

	CUDA_ERR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUSPARSE_ERR(cusparseCreate(&handle));
	CUSPARSE_ERR(cusparseSetStream(handle, stream));

	//
	// ---- Step 3. Allocate Buffer ---- 
	CUSPARSE_ERR(
		cusparseXcoosort_bufferSizeExt(
			handle,
			M, N, NZ,
			device_JR, device_JC, &buffer_size
		)
	);

	CUDA_ERR(cudaMalloc((void**)&device_JR, sizeof(int) * NZ));
	CUDA_ERR(cudaMalloc((void**)&device_JC, sizeof(int) * NZ));
	CUDA_ERR(cudaMalloc((void**)&device_P, sizeof(int) * NZ));
	CUDA_ERR(cudaMalloc((void**)&device_AA, sizeof(float) * NZ));
	CUDA_ERR(cudaMalloc((void**)&device_AA_sorted, sizeof(float) * NZ));
	CUDA_ERR(cudaMalloc((void**)&buffer, sizeof(char) * buffer_size));

	CUDA_ERR(cudaMemcpy(device_JR, host_JR, sizeof(int) * NZ, cudaMemcpyHostToDevice));
	CUDA_ERR(cudaMemcpy(device_JC, host_JC, sizeof(int) * NZ, cudaMemcpyHostToDevice));
	CUDA_ERR(cudaMemcpy(device_AA, host_AA, sizeof(float) * NZ, cudaMemcpyHostToDevice));
	CUDA_ERR(cudaDeviceSynchronize());

	//
	// ---- Step 4. Setup permutation vector P to Identity ---- 
	CUSPARSE_ERR(cusparseCreateIdentityPermutation(handle, NZ, device_P));

	//
	// ---- Step 5. Sort ---- 
	CUSPARSE_ERR(
		cusparseXcoosortByRow(handle, M, N, NZ, device_JR, device_JC, device_P, buffer)
	);

	// Gather
	// CUSPARSE_ERR(cusparseDgthr(
	//	handle, NZ, device_AA, device_AA_sorted, device_P, CUSPARSE_INDEX_BASE_ZERO));
	CUSPARSE_ERR(cusparseSgthr(
		handle, NZ, device_AA, device_AA_sorted, device_P, CUSPARSE_INDEX_BASE_ZERO));
	CUDA_ERR(cudaDeviceSynchronize());

	// Fetch back
	CUDA_ERR(cudaMemcpy(host_JR, device_JR, sizeof(int) * NZ, cudaMemcpyDeviceToHost));
	CUDA_ERR(cudaMemcpy(host_JC, device_JC, sizeof(int) * NZ, cudaMemcpyDeviceToHost));
	CUDA_ERR(cudaMemcpy(host_P, device_P, sizeof(int) * NZ, cudaMemcpyDeviceToHost));
	CUDA_ERR(cudaMemcpy(host_AA, device_AA_sorted, sizeof(float) * NZ, cudaMemcpyDeviceToHost));
	CUDA_ERR(cudaDeviceSynchronize());

	// ---- Step 6. Free resources ---- 
#ifdef CUSPARSE_CSR
	if (device_JR) cudaFree(device_JR);
	if (device_JC) cudaFree(device_JC);
#endif
	if (device_P) cudaFree(device_P);
	if (device_AA) cudaFree(device_AA);
	if (buffer) cudaFree(buffer);
	if (handle) cusparseDestroy(handle);
	if (stream) cudaStreamDestroy(stream);

	free(host_P); // Unnecessary


#if defined( CUSPARSE_CSR )
	int* t_JR	= (int*)calloc((M + 1), sizeof(int));
	int* t_JC	= (int*)malloc(NZ * sizeof(int));
	float* t_AA = (float*)malloc(NZ * sizeof(float));
	for (int i = 0; i < M + 1; i++) t_JR[i]++;

	for (int i = 0; i < NZ; i++) {
		t_AA[i] = host_AA[i];
		t_JC[i] = host_JC[i];
		t_JR[host_JR[i]]++;
	}

	for (int i = 0; i < M; i++)	t_JR[i + 1] += (t_JR[i] - 1);

	free(host_JR);
	free(host_JC);
	free(host_AA);

	host_JR = t_JR;
	host_JC = t_JC;
	host_AA = t_AA;

#endif


	// ----               ----
	// ---- cuSPARSE SpMV ----
	// ----               ----
	handle		= NULL;
	buffer		= NULL;
	buffer_size = 0;

	float elapsed = 0;
	cudaEvent_t start, stop;

#ifdef CUSPARSE
	{ // SpMV
		printf("\n#### \tSpMV cuSPARSE \t####\n");
		// ---- Step 7. Define variables
		const float alpha	= 1;
		const float beta	= 0;

		float host_y[N]		= {0, };
		float host_x[M];

		float* device_x		= NULL;
		float* device_y		= NULL;

		for (auto& elem : host_x) elem = 1;

		cusparseSpMatDescr_t sp_mtx; // device
		cusparseDnVecDescr_t dn_x, dn_y; // device

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start); // Timer start

		// ---- Step 8. Get your GPU memory ready ----
		CUDA_ERR(cudaMalloc((void**)&device_x, sizeof(float) * M));
		CUDA_ERR(cudaMalloc((void**)&device_y, sizeof(float) * N));

		CUDA_ERR(cudaMemcpy(device_x, host_x, sizeof(float) * M, cudaMemcpyHostToDevice));
		CUDA_ERR(cudaMemcpy(device_y, host_y, sizeof(float) * N, cudaMemcpyHostToDevice));

#ifdef CUSPARSE_CSR
		CUDA_ERR(cudaMalloc((void**)&device_JR, sizeof(int) * (M + 1)));
		CUDA_ERR(cudaMalloc((void**)&device_JC, sizeof(int) * NZ));

		CUDA_ERR(cudaMemcpy(device_JR, host_JR, sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
		CUDA_ERR(cudaMemcpy(device_JC, host_JC, sizeof(int) * NZ, cudaMemcpyHostToDevice));
#endif

		CUSPARSE_ERR(cusparseCreate(&handle));

		// Create sparse matrix
#ifndef CUSPARSE_CSR
		CUSPARSE_ERR(
			cusparseCreateCoo(
				&sp_mtx, 
				M, N, NZ, device_JR, device_JC, device_AA_sorted,
				CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F)
		);
#else
		CUSPARSE_ERR(
			cusparseCreateCsr(
				&sp_mtx,
				M, N, NZ, device_JR, device_JC, device_AA_sorted,
				CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F)
		);
#endif
		
		CUSPARSE_ERR(cusparseCreateDnVec(&dn_x, N, device_x, CUDA_R_32F));
		CUSPARSE_ERR(cusparseCreateDnVec(&dn_y, M, device_y, CUDA_R_32F));

#ifndef CUSPARSE_CSR
		CUSPARSE_ERR(cusparseSpMV_bufferSize(
			handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, sp_mtx, dn_x, &beta, dn_y, CUDA_R_32F,
			CUSPARSE_COOMV_ALG, &buffer_size));
#else
		CUSPARSE_ERR(cusparseSpMV_bufferSize(
			handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, sp_mtx, dn_x, &beta, dn_y, CUDA_R_32F,
			CUSPARSE_CSRMV_ALG1, &buffer_size));
#endif

		CUDA_ERR(cudaMalloc(&buffer, buffer_size));
		
		// ---- Step 9. Do SpMV ----
#ifndef CUSPARSE_CSR
		CUSPARSE_ERR(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, sp_mtx, dn_x, &beta, dn_y, CUDA_R_32F,
			CUSPARSE_COOMV_ALG, buffer));
#else
		CUSPARSE_ERR(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, sp_mtx, dn_x, &beta, dn_y, CUDA_R_32F,
			CUSPARSE_CSRMV_ALG1, buffer));


		// ---- Step 11. Destroy ----
		CUSPARSE_ERR(cusparseDestroySpMat(sp_mtx));
		CUSPARSE_ERR(cusparseDestroyDnVec(dn_x));
		CUSPARSE_ERR(cusparseDestroyDnVec(dn_y));
#endif

		// ---- Step 10. Fetch the result ----
		CUDA_ERR(cudaMemcpy(host_y, device_y, N * sizeof(float), cudaMemcpyDeviceToHost));

		// Record
		cudaEventRecord(stop); // timer end
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);

		for (int i = 0; i < 10; i++) 
			printf("%7.1f", host_y[i]); // Check		

		// ---- Step 12. Return resources ----
		if (device_JR) cudaFree(device_JR);
		if (device_JC) cudaFree(device_JC);
		if (device_AA_sorted) cudaFree(device_AA_sorted);
		if (device_x) cudaFree(device_x);
		if (device_y) cudaFree(device_y);
		if (buffer) cudaFree(buffer);
		if (handle) cusparseDestroy(handle);


		printf("\nElapsed: %fms\n", elapsed);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}	
#else

	// ----             ----
	// ---- Kernel SpMV ----
	// ----             ----
	{
		printf("\n#### \tSpMV Kernel \t####\n");
		float host_y[N]		= { 0, };
		float host_x[M];

		float* device_x		= NULL;
		float* device_y		= NULL;

		for (auto& elem : host_x) elem = 1;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start); // Timer start

		CUDA_ERR(cudaMalloc((void**)&device_JR, sizeof(int) * (M + 1)));
		CUDA_ERR(cudaMalloc((void**)&device_JC, sizeof(int) * NZ));
		CUDA_ERR(cudaMalloc((void**)&device_x, sizeof(float) * M));
		CUDA_ERR(cudaMalloc((void**)&device_y, sizeof(float) * N));

		CUDA_ERR(cudaMemcpy(device_JR, host_JR, sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
		CUDA_ERR(cudaMemcpy(device_JC, host_JC, sizeof(int) * NZ, cudaMemcpyHostToDevice));
		CUDA_ERR(cudaMemcpy(device_x, host_x, sizeof(float) * M, cudaMemcpyHostToDevice));
		CUDA_ERR(cudaMemcpy(device_y, host_y, sizeof(float) * N, cudaMemcpyHostToDevice));

#ifdef KERNEL_CSR_SCALAR
#ifdef CASE1
		ker_csr_spmv_scalar<<<1, M>>>(
#endif
#ifdef CASE2
		ker_csr_spmv_scalar<<<1, M>>>(
#endif
#ifdef CASE3
		ker_csr_spmv_scalar<<<2, M / 2>>>(
#endif
#ifdef CASE4
		ker_csr_spmv_scalar<<<4, M / 4>>>(
#endif
				device_JR, device_JC, device_AA_sorted, device_x, device_y
			);
		// cudaDeviceSynchronize();
#endif
#ifdef KERNEL_CSR_VECTOR
#ifdef CASE1
		ker_csr_spmv_vector <<<1, 32 * M>>>(
#endif
#ifdef CASE2
		ker_csr_spmv_vector <<<32, M>>>(
#endif
#ifdef CASE3
		ker_csr_spmv_vector <<<64, 1024>>>(
#endif
#ifdef CASE4
		ker_csr_spmv_vector <<<128, 1024>>>(
#endif
			device_JR, device_JC, device_AA_sorted, device_x, device_y
		);
#endif

		// ---- Step 10. Fetch the result ----
		CUDA_ERR(cudaMemcpy(host_y, device_y, N * sizeof(float), cudaMemcpyDeviceToHost));

		// Record
		cudaEventRecord(stop); // timer end
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);

		for (int i = 0; i < 10; i++)
			printf("%7.1f", host_y[i]); // Check		

		if (device_JR) cudaFree(device_JR);
		if (device_JC) cudaFree(device_JC);
		if (device_AA_sorted) cudaFree(device_AA_sorted);
		if (device_x) cudaFree(device_x);
		if (device_y) cudaFree(device_y);
		if (buffer) cudaFree(buffer);
		if (handle) cusparseDestroy(handle);

		printf("\nElapsed: %fms\n", elapsed);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

#endif

	free(host_JR);
	free(host_JC);
	free(host_AA);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	if (cudaDeviceReset() != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}