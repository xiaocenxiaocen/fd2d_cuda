/* @file: fd2d_gpu_kernel.cu
 * @author: Zhang Xiao
 * @date: 2016.09.18
 */
#include <cstdio>
#include <math.h>
#include <omp.h>
#include <assert.h>

#include <cuda_runtime.h>
//#include <helper_cuda.h>
//#include <helper_functions.h>

#define fd8_b0 -2.847222222
#define fd8_b1 1.6
#define fd8_b2 -0.2
#define fd8_b3 0.02539682540
#define fd8_b4 -0.001785714286

#define fd8_a1 0.8
#define fd8_a2 -0.2
#define fd8_a3 0.038095238
#define fd8_a4 -0.0035714286

#define SCHEME 8
#define HALF_SCHEME (SCHEME / 2)
#define BLOCK_X 32
#define BLOCK_Y 8
#define STENCIL_X BLOCK_X + 2 * HALF_SCHEME
#define STENCIL_Y BLOCK_Y + 2 * HALF_SCHEME

#define R 1e-8

enum PML2D_Type{X, Z};

class pml2d_gpu_t {
public:
	pml2d_gpu_t(const int nx, const int nz, PML2D_Type pml2d_type): nx(nx), nz(nz) { length = pml2d_type == X ? BLOCK_X * nz : nx * nz; };
	~pml2d_gpu_t();
	void pml2d_gpu_allocate();
	inline void pml2d_init(const int pml_idx, const float damp_x, const float damp_dx, const float alpha_x, const float alpha_dx, const float dt);
	void pml2d_memcpyh2d();
//	inline __device__ void pml2d_gpu_update(const int pml_idx, const float vel, const float ux, const float uxx);
public:
	float * h_ax;
	float * h_bx;
	float * h_dadx;
	float * h_dbdx;
	float * d_ax;
	float * d_bx;
	float * d_dadx;
	float * d_dbdx;
	float * d_psi;
	float * d_eta;
	float * d_theta;
	float * d_uxx;
private:
	int nx;
	int nz;
	size_t length;
};

pml2d_gpu_t::~pml2d_gpu_t()
{
	cudaFree(d_ax);
	cudaFree(d_bx);
	cudaFree(d_dadx);
	cudaFree(d_dbdx);
	cudaFree(d_psi);
	cudaFree(d_eta);
	cudaFree(d_theta);

	cudaFree(d_uxx);
}

void pml2d_gpu_t::pml2d_gpu_allocate()
{
	h_ax = (float*)malloc(sizeof(float) * length);
	h_bx = (float*)malloc(sizeof(float) * length);
	h_dadx = (float*)malloc(sizeof(float) * length);
	h_dbdx = (float*)malloc(sizeof(float) * length);

	memset((void*)h_ax, 0, sizeof(float) * length);
	memset((void*)h_bx, 0, sizeof(float) * length);
	memset((void*)h_dadx, 0, sizeof(float) * length);
	memset((void*)h_dbdx, 0, sizeof(float) * length);

	cudaMalloc((void**)&d_ax, sizeof(float) * length);
	cudaMalloc((void**)&d_bx, sizeof(float) * length);
	cudaMalloc((void**)&d_dadx, sizeof(float) * length);
	cudaMalloc((void**)&d_dbdx, sizeof(float) * length);
	cudaMalloc((void**)&d_psi, sizeof(float) * length);
	cudaMalloc((void**)&d_eta, sizeof(float) * length);
	cudaMalloc((void**)&d_theta, sizeof(float) * length);

	cudaMemset(d_ax, 0, sizeof(float) * length);
	cudaMemset(d_bx, 0, sizeof(float) * length);
	cudaMemset(d_dadx, 0, sizeof(float) * length);
	cudaMemset(d_dbdx, 0, sizeof(float) * length);
	cudaMemset(d_psi, 0, sizeof(float) * length);
	cudaMemset(d_eta, 0, sizeof(float) * length);
	cudaMemset(d_theta, 0, sizeof(float) * length);

	cudaMalloc((void**)&d_uxx, sizeof(float) * length);
	cudaMemset(d_uxx, 0, sizeof(float) * length);

	return;
}

inline void pml2d_gpu_t::pml2d_init(const int pml_idx, const float damp_x, const float damp_dx, const float alpha_x, const float alpha_dx, const float dt)
{
	h_bx[pml_idx] = expf(-(damp_x + alpha_x) * dt);
	h_ax[pml_idx] = (h_bx[pml_idx] - 1.0f) * damp_x / (damp_x + alpha_x);
	h_dbdx[pml_idx] = h_bx[pml_idx] * (-damp_dx - alpha_dx) * dt;
	h_dadx[pml_idx] = damp_dx/(damp_x + alpha_x) * (h_bx[pml_idx] - 1.0f) - (damp_dx + alpha_dx)/(damp_x + alpha_x)*h_ax[pml_idx] + damp_x/(damp_x + alpha_x) * h_dbdx[pml_idx];
}

void pml2d_gpu_t::pml2d_memcpyh2d()
{
	cudaMemcpy(d_ax, h_ax, sizeof(float) * length, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bx, h_bx, sizeof(float) * length, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dadx, h_dadx, sizeof(float) * length, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dbdx, h_dbdx, sizeof(float) * length, cudaMemcpyHostToDevice);
	
	free(h_ax); h_ax = NULL;
	free(h_bx); h_bx = NULL;
	free(h_dadx); h_dadx = NULL;
	free(h_dbdx); h_dbdx = NULL;

	return;
}

inline __device__ void pml2d_gpu_update(const int pml_idx, const float vel, const float ux, const float uxx, pml2d_gpu_t * pml)
{
	float tmp = pml->d_theta[pml_idx];
	float bx = pml->d_bx[pml_idx];
	float ax = pml->d_ax[pml_idx];
	float dbdx = pml->d_dbdx[pml_idx];
	float dadx = pml->d_dadx[pml_idx];

	float psiTmp = pml->d_psi[pml_idx];	

	float thetaTmp; 
	pml->d_theta[pml_idx] = thetaTmp = bx * tmp + ax * uxx + dbdx * psiTmp + dadx * ux;
	
	tmp = 0.5f * thetaTmp + 0.5f * tmp;

	float etaTmp;	
	pml->d_eta[pml_idx] = etaTmp = bx * pml->d_eta[pml_idx] + ax * (uxx + tmp);

	pml->d_psi[pml_idx] = bx * psiTmp + ax * ux;

	pml->d_uxx[pml_idx] = vel * (thetaTmp + etaTmp);
}

// assert 2 * HALF_SCHEME < blockDim.y = npmlz1 - HALF_SCHEME
#define NPMLZ 12
#define NPMLX 12
#define BLOCK_PML_Y NPMLZ - HALF_SCHEME
#define BLOCK_PML_X BLOCK_X
#define STENCIL_PML_Y BLOCK_PML_Y + 2 * HALF_SCHEME
__global__ static void ZTopPML2DUpdateKernel
(
	float * __restrict__ um,
	float * __restrict__ uo,
	float * __restrict__ vel,
	const int lda,
	const float dx, 
	const float dz,
	pml2d_gpu_t * pml
)
{
	__shared__ float smem[STENCIL_PML_Y][BLOCK_X];

	const int thrdBgnX = HALF_SCHEME + blockIdx.x * blockDim.x;
	const int thrdBgnY = blockIdx.y * blockDim.y;
	
	const int ix = thrdBgnX + threadIdx.x;
	const int iy = thrdBgnY + threadIdx.y;
	smem[threadIdx.y][threadIdx.x] = uo[iy * lda + ix];
	if(threadIdx.y < 2 * HALF_SCHEME) {
		smem[threadIdx.y + blockDim.y][threadIdx.x] = uo[(iy + blockDim.y) * lda + ix];
	}
	
	__syncthreads();

	float uz, uzz;
	uzz  = fd8_b4 * smem[threadIdx.y + HALF_SCHEME - 4][threadIdx.x];
	uzz += fd8_b3 * smem[threadIdx.y + HALF_SCHEME - 3][threadIdx.x];
	uzz += fd8_b2 * smem[threadIdx.y + HALF_SCHEME - 2][threadIdx.x];
	uzz += fd8_b1 * smem[threadIdx.y + HALF_SCHEME - 1][threadIdx.x];
	uzz += fd8_b0 * smem[threadIdx.y + HALF_SCHEME    ][threadIdx.x];
	uzz += fd8_b1 * smem[threadIdx.y + HALF_SCHEME + 1][threadIdx.x];
	uzz += fd8_b2 * smem[threadIdx.y + HALF_SCHEME + 2][threadIdx.x];
	uzz += fd8_b3 * smem[threadIdx.y + HALF_SCHEME + 3][threadIdx.x];
	uzz += fd8_b4 * smem[threadIdx.y + HALF_SCHEME + 4][threadIdx.x];

	uz  = - fd8_a4 * smem[threadIdx.y + HALF_SCHEME - 4][threadIdx.x];
	uz += - fd8_a3 * smem[threadIdx.y + HALF_SCHEME - 3][threadIdx.x];
	uz += - fd8_a2 * smem[threadIdx.y + HALF_SCHEME - 2][threadIdx.x];
	uz += - fd8_a1 * smem[threadIdx.y + HALF_SCHEME - 1][threadIdx.x];
	uz +=   fd8_a1 * smem[threadIdx.y + HALF_SCHEME + 1][threadIdx.x];
	uz +=   fd8_a2 * smem[threadIdx.y + HALF_SCHEME + 2][threadIdx.x];
	uz +=   fd8_a3 * smem[threadIdx.y + HALF_SCHEME + 3][threadIdx.x];
	uz +=   fd8_a4 * smem[threadIdx.y + HALF_SCHEME + 4][threadIdx.x];

	__syncthreads();

	smem[threadIdx.y][threadIdx.x] = um[iy * lda + ix];
	if(threadIdx.y < 2 * HALF_SCHEME) {
		smem[threadIdx.y + blockDim.y][threadIdx.x] = um[(iy + blockDim.y) * lda + ix];
	}
	
	__syncthreads();	

	uzz += fd8_b4 * smem[threadIdx.y + HALF_SCHEME - 4][threadIdx.x];
	uzz += fd8_b3 * smem[threadIdx.y + HALF_SCHEME - 3][threadIdx.x];
	uzz += fd8_b2 * smem[threadIdx.y + HALF_SCHEME - 2][threadIdx.x];
	uzz += fd8_b1 * smem[threadIdx.y + HALF_SCHEME - 1][threadIdx.x];
	uzz += fd8_b0 * smem[threadIdx.y + HALF_SCHEME    ][threadIdx.x];
	uzz += fd8_b1 * smem[threadIdx.y + HALF_SCHEME + 1][threadIdx.x];
	uzz += fd8_b2 * smem[threadIdx.y + HALF_SCHEME + 2][threadIdx.x];
	uzz += fd8_b3 * smem[threadIdx.y + HALF_SCHEME + 3][threadIdx.x];
	uzz += fd8_b4 * smem[threadIdx.y + HALF_SCHEME + 4][threadIdx.x];

	uz += - fd8_a4 * smem[threadIdx.y + HALF_SCHEME - 4][threadIdx.x];
	uz += - fd8_a3 * smem[threadIdx.y + HALF_SCHEME - 3][threadIdx.x];
	uz += - fd8_a2 * smem[threadIdx.y + HALF_SCHEME - 2][threadIdx.x];
	uz += - fd8_a1 * smem[threadIdx.y + HALF_SCHEME - 1][threadIdx.x];
	uz +=   fd8_a1 * smem[threadIdx.y + HALF_SCHEME + 1][threadIdx.x];
	uz +=   fd8_a2 * smem[threadIdx.y + HALF_SCHEME + 2][threadIdx.x];
	uz +=   fd8_a3 * smem[threadIdx.y + HALF_SCHEME + 3][threadIdx.x];
	uz +=   fd8_a4 * smem[threadIdx.y + HALF_SCHEME + 4][threadIdx.x];

	uzz /= (2.0f * dz * dz );
	uz /= 2.0f * dz;

	const int pmlIdx = (iy + HALF_SCHEME) * lda + ix;
	const int gloIdx = (iy + HALF_SCHEME) * lda + ix;
	pml2d_gpu_update(pmlIdx, vel[gloIdx], uz, uzz, pml);
}

__global__ static void ZTopPML2DUpdateWavefieldKernel
(
	float * __restrict__ out,
	pml2d_gpu_t * pml,
	const int lda
)
{
	const int thrdBgnX = HALF_SCHEME + blockIdx.x * blockDim.x;
	const int thrdBgnY = HALF_SCHEME + blockIdx.y * blockDim.y;
	
	const int ix = thrdBgnX + threadIdx.x;
	const int iy = thrdBgnY + threadIdx.y;

	const int pmlIdx = iy * lda + ix;
	const int gloIdx = iy * lda + ix;
	out[gloIdx] += pml->d_uxx[pmlIdx];
}

__global__ static void ZBottomPML2DUpdateKernel
(
	float * __restrict__ um,
	float * __restrict__ uo,
	float * __restrict__ vel,
	const int lda,
	const float dx, 
	const float dz,
	const int nzPad, 
	pml2d_gpu_t * pml
)
{
	__shared__ float smem[STENCIL_PML_Y][BLOCK_X];

	const int thrdBgnX = HALF_SCHEME + blockIdx.x * blockDim.x;
	const int thrdBgnY = nzPad - NPMLZ - HALF_SCHEME;
	
	const int ix = thrdBgnX + threadIdx.x;
	const int iy = thrdBgnY + threadIdx.y;
	smem[threadIdx.y][threadIdx.x] = uo[iy * lda + ix];
	if(threadIdx.y < 2 * HALF_SCHEME) {
		smem[threadIdx.y + blockDim.y][threadIdx.x] = uo[(iy + blockDim.y) * lda + ix];
	}
	
	__syncthreads();

	float uz, uzz;
	uzz  = fd8_b4 * smem[threadIdx.y + HALF_SCHEME - 4][threadIdx.x];
	uzz += fd8_b3 * smem[threadIdx.y + HALF_SCHEME - 3][threadIdx.x];
	uzz += fd8_b2 * smem[threadIdx.y + HALF_SCHEME - 2][threadIdx.x];
	uzz += fd8_b1 * smem[threadIdx.y + HALF_SCHEME - 1][threadIdx.x];
	uzz += fd8_b0 * smem[threadIdx.y + HALF_SCHEME    ][threadIdx.x];
	uzz += fd8_b1 * smem[threadIdx.y + HALF_SCHEME + 1][threadIdx.x];
	uzz += fd8_b2 * smem[threadIdx.y + HALF_SCHEME + 2][threadIdx.x];
	uzz += fd8_b3 * smem[threadIdx.y + HALF_SCHEME + 3][threadIdx.x];
	uzz += fd8_b4 * smem[threadIdx.y + HALF_SCHEME + 4][threadIdx.x];

	uz  = - fd8_a4 * smem[threadIdx.y + HALF_SCHEME - 4][threadIdx.x];
	uz += - fd8_a3 * smem[threadIdx.y + HALF_SCHEME - 3][threadIdx.x];
	uz += - fd8_a2 * smem[threadIdx.y + HALF_SCHEME - 2][threadIdx.x];
	uz += - fd8_a1 * smem[threadIdx.y + HALF_SCHEME - 1][threadIdx.x];
	uz +=   fd8_a1 * smem[threadIdx.y + HALF_SCHEME + 1][threadIdx.x];
	uz +=   fd8_a2 * smem[threadIdx.y + HALF_SCHEME + 2][threadIdx.x];
	uz +=   fd8_a3 * smem[threadIdx.y + HALF_SCHEME + 3][threadIdx.x];
	uz +=   fd8_a4 * smem[threadIdx.y + HALF_SCHEME + 4][threadIdx.x];

	__syncthreads();

	smem[threadIdx.y][threadIdx.x] = um[iy * lda + ix];
	if(threadIdx.y < 2 * HALF_SCHEME) {
		smem[threadIdx.y + blockDim.y][threadIdx.x] = um[(iy + blockDim.y) * lda + ix];
	}
	
	__syncthreads();	

	uzz += fd8_b4 * smem[threadIdx.y + HALF_SCHEME - 4][threadIdx.x];
	uzz += fd8_b3 * smem[threadIdx.y + HALF_SCHEME - 3][threadIdx.x];
	uzz += fd8_b2 * smem[threadIdx.y + HALF_SCHEME - 2][threadIdx.x];
	uzz += fd8_b1 * smem[threadIdx.y + HALF_SCHEME - 1][threadIdx.x];
	uzz += fd8_b0 * smem[threadIdx.y + HALF_SCHEME    ][threadIdx.x];
	uzz += fd8_b1 * smem[threadIdx.y + HALF_SCHEME + 1][threadIdx.x];
	uzz += fd8_b2 * smem[threadIdx.y + HALF_SCHEME + 2][threadIdx.x];
	uzz += fd8_b3 * smem[threadIdx.y + HALF_SCHEME + 3][threadIdx.x];
	uzz += fd8_b4 * smem[threadIdx.y + HALF_SCHEME + 4][threadIdx.x];

	uz += - fd8_a4 * smem[threadIdx.y + HALF_SCHEME - 4][threadIdx.x];
	uz += - fd8_a3 * smem[threadIdx.y + HALF_SCHEME - 3][threadIdx.x];
	uz += - fd8_a2 * smem[threadIdx.y + HALF_SCHEME - 2][threadIdx.x];
	uz += - fd8_a1 * smem[threadIdx.y + HALF_SCHEME - 1][threadIdx.x];
	uz +=   fd8_a1 * smem[threadIdx.y + HALF_SCHEME + 1][threadIdx.x];
	uz +=   fd8_a2 * smem[threadIdx.y + HALF_SCHEME + 2][threadIdx.x];
	uz +=   fd8_a3 * smem[threadIdx.y + HALF_SCHEME + 3][threadIdx.x];
	uz +=   fd8_a4 * smem[threadIdx.y + HALF_SCHEME + 4][threadIdx.x];

	uzz /= (2.0f * dz * dz );
	uz /= 2.0f * dz;

	const int pmlIdx = threadIdx.y * lda + ix;
	const int gloIdx = (iy + HALF_SCHEME) * lda + ix;
	pml2d_gpu_update(pmlIdx, vel[gloIdx], uz, uzz, pml);
}

__global__ static void ZBottomPML2DUpdateWavefieldKernel
(
	float * __restrict__ out,
	pml2d_gpu_t * pml,
	const int lda, 
	const int nzPad
)
{
	const int thrdBgnX = HALF_SCHEME + blockIdx.x * blockDim.x;
	const int thrdBgnY = nzPad - NPMLZ;
	
	const int ix = thrdBgnX + threadIdx.x;
	const int iy = thrdBgnY + threadIdx.y;

	const int pmlIdx = threadIdx.y * lda + ix;
	const int gloIdx = iy * lda + ix;
	out[gloIdx] += pml->d_uxx[pmlIdx];
}

void PrintDeviceInfo()
{	
	int devCount;

	//checkCudaErrors(cudaGetDeviceCount(&devCount));
	cudaGetDeviceCount(&devCount);

	if(devCount==0) {
		fprintf(stderr, "There is no device supporting CUDA\n");
	} else {

		for(int dev=0;dev<devCount;dev++) {
			cudaDeviceProp deviceProp;
			
			//checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
			cudaGetDeviceProperties(&deviceProp, dev);

			if(dev==0) {
				if(deviceProp.major==9999 && deviceProp.minor==9999) {
					fprintf(stderr, "There is no device supporting CUDA\n");
				} else {
					if(devCount==1) {
						fprintf(stderr, "There is 1 device supporting CUDA\n");
					} else {
						fprintf(stderr, "There are %d devices supporting CUDA\n", devCount);
					}
				}
	
			} // end if
	
			fprintf(stderr, "\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	
			fprintf(stderr, "  Major revision number:                               %d\n", deviceProp.major);
			fprintf(stderr, "  Minor revision number:                               %d\n", deviceProp.minor);
			fprintf(stderr, "  Total amount of global memory:                       %dG bytes\n", deviceProp.totalGlobalMem/(1024*1024*1024));
			fprintf(stderr, "  Number of multiprocessors:                           %d\n", deviceProp.multiProcessorCount);
			fprintf(stderr, "  Number of cores:                                     %d\n", deviceProp.multiProcessorCount*8);
			fprintf(stderr, "  Total amount of constant memory:                     %dK bytes\n", deviceProp.totalConstMem/1024);
			fprintf(stderr, "  Total amount of shared memory per block:             %dK bytes\n", deviceProp.sharedMemPerBlock/1024);
			fprintf(stderr, "  Total number of register available per block:        %d\n", deviceProp.regsPerBlock);
			fprintf(stderr, "  Warp size:                                           %d\n", deviceProp.warpSize);
			fprintf(stderr, "  Maximum number of threads per block:                 %d\n", deviceProp.maxThreadsPerBlock);
			fprintf(stderr, "  Maximum sizes of each dimension of a block:          %d x %d x %d\n", 
					deviceProp.maxThreadsDim[0],
					deviceProp.maxThreadsDim[1],
					deviceProp.maxThreadsDim[2]);
			fprintf(stderr, "  Maximum sizes of each dimension of a grid:           %d x %d x %d\n",
					deviceProp.maxGridSize[0],
					deviceProp.maxGridSize[1],
					deviceProp.maxGridSize[2]);
			fprintf(stderr, "  Maximum memory pitch:                                %u bytes\n", deviceProp.memPitch);
			fprintf(stderr, "  Texture alignment:                                   %u bytes\n", deviceProp.textureAlignment);
			fprintf(stderr, "  Clock rate:                                          %.2f GHz\n", deviceProp.clockRate*1e-6f);
			fprintf(stderr, "  Concurrent copy and execution:                       %s\n", deviceProp.deviceOverlap ? "Yes":"No");

		} // end for

		fprintf(stderr, "\nTest PASSED\n");
	} // end if
}

// assert BLOCK_X >= NPMLX + HALF_SCHEME
__global__ static void XLeftPML2DUpdateKernel
(
	float * __restrict__ um,
	float * __restrict__ uo,
	float * __restrict__ vel,
	const int lda,
	const float dx, 
	const float dz,
	pml2d_gpu_t * pml
)
{
	__shared__ float smem[BLOCK_Y][BLOCK_X];

	const int thrdBgnX = blockIdx.x * blockDim.x;
	const int thrdBgnY = HALF_SCHEME + blockIdx.y * blockDim.y;
	
	const int ix = thrdBgnX + threadIdx.x;
	const int iy = thrdBgnY + threadIdx.y;
	smem[threadIdx.y][threadIdx.x] = uo[iy * lda + ix];
	
	__syncthreads();

	float ux, uxx;

	if(threadIdx.x < NPMLX - HALF_SCHEME) {
		uxx  = fd8_b4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 4];
		uxx += fd8_b3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 3];
		uxx += fd8_b2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 2];
		uxx += fd8_b1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 1];
		uxx += fd8_b0 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME    ];
		uxx += fd8_b1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 1];
		uxx += fd8_b2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 2];
		uxx += fd8_b3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 3];
		uxx += fd8_b4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 4];
	
		ux  = - fd8_a4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 4];
		ux += - fd8_a3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 3];
		ux += - fd8_a2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 2];
		ux += - fd8_a1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 1];
		ux +=   fd8_a1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 1];
		ux +=   fd8_a2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 2];
		ux +=   fd8_a3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 3];
		ux +=   fd8_a4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 4];
	}
	
	__syncthreads();

	smem[threadIdx.y][threadIdx.x] = um[iy * lda + ix];
	
	__syncthreads();
	
	if(threadIdx.x < NPMLX - HALF_SCHEME) {
		uxx += fd8_b4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 4];
		uxx += fd8_b3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 3];
		uxx += fd8_b2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 2];
		uxx += fd8_b1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 1];
		uxx += fd8_b0 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME    ];
		uxx += fd8_b1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 1];
		uxx += fd8_b2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 2];
		uxx += fd8_b3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 3];
		uxx += fd8_b4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 4];
	
		ux += - fd8_a4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 4];
		ux += - fd8_a3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 3];
		ux += - fd8_a2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 2];
		ux += - fd8_a1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 1];
		ux +=   fd8_a1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 1];
		ux +=   fd8_a2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 2];
		ux +=   fd8_a3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 3];
		ux +=   fd8_a4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 4];
	}

	uxx /= (2.0f * dx * dx );
	ux /= 2.0f * dx;

	const int pmlIdx = iy * BLOCK_X + ix + HALF_SCHEME;
	const int gloIdx = iy * lda + ix + HALF_SCHEME;
	
	if(threadIdx.x < NPMLX - HALF_SCHEME) pml2d_gpu_update(pmlIdx, vel[gloIdx], ux, uxx, pml);
}

__global__ static void XLeftPML2DUpdateWavefieldKernel
(
	float * __restrict__ out,
	pml2d_gpu_t * pml,
	const int lda
)
{
	const int thrdBgnX = HALF_SCHEME + blockIdx.x * blockDim.x;
	const int thrdBgnY = HALF_SCHEME + blockIdx.y * blockDim.y;
	
	const int ix = thrdBgnX + threadIdx.x;
	const int iy = thrdBgnY + threadIdx.y;

	const int pmlIdx = iy * BLOCK_X + ix;
	const int gloIdx = iy * lda + ix;

	if(threadIdx.x < NPMLX - HALF_SCHEME) out[gloIdx] += pml->d_uxx[pmlIdx];
}

__global__ static void XRightPML2DUpdateKernel
(
	float * __restrict__ um,
	float * __restrict__ uo,
	float * __restrict__ vel,
	const int lda,
	const float dx, 
	const float dz,
	const int nxPad, 
	pml2d_gpu_t * pml
)
{
	__shared__ float smem[BLOCK_Y][BLOCK_X];

	const int thrdBgnX = nxPad - NPMLX - HALF_SCHEME;
	const int thrdBgnY = HALF_SCHEME + blockIdx.y * blockDim.y;
	
	const int ix = thrdBgnX + threadIdx.x;
	const int iy = thrdBgnY + threadIdx.y;
	if(threadIdx.x < NPMLX + HALF_SCHEME) {
		smem[threadIdx.y][threadIdx.x] = uo[iy * lda + ix];
	}
	
	__syncthreads();

	float ux, uxx;

	if(threadIdx.x < NPMLX - HALF_SCHEME) {
		uxx  = fd8_b4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 4];
		uxx += fd8_b3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 3];
		uxx += fd8_b2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 2];
		uxx += fd8_b1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 1];
		uxx += fd8_b0 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME    ];
		uxx += fd8_b1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 1];
		uxx += fd8_b2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 2];
		uxx += fd8_b3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 3];
		uxx += fd8_b4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 4];
	
		ux  = - fd8_a4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 4];
		ux += - fd8_a3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 3];
		ux += - fd8_a2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 2];
		ux += - fd8_a1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 1];
		ux +=   fd8_a1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 1];
		ux +=   fd8_a2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 2];
		ux +=   fd8_a3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 3];
		ux +=   fd8_a4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 4];
	}
	
	__syncthreads();

	if(threadIdx.x < NPMLX + HALF_SCHEME) {
		smem[threadIdx.y][threadIdx.x] = um[iy * lda + ix];
	}
	
	__syncthreads();
	
	if(threadIdx.x < NPMLX - HALF_SCHEME) {
		uxx += fd8_b4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 4];
		uxx += fd8_b3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 3];
		uxx += fd8_b2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 2];
		uxx += fd8_b1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 1];
		uxx += fd8_b0 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME    ];
		uxx += fd8_b1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 1];
		uxx += fd8_b2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 2];
		uxx += fd8_b3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 3];
		uxx += fd8_b4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 4];
	
		ux += - fd8_a4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 4];
		ux += - fd8_a3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 3];
		ux += - fd8_a2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 2];
		ux += - fd8_a1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME - 1];
		ux +=   fd8_a1 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 1];
		ux +=   fd8_a2 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 2];
		ux +=   fd8_a3 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 3];
		ux +=   fd8_a4 * smem[threadIdx.y][threadIdx.x + HALF_SCHEME + 4];
	}

	uxx /= (2.0f * dx * dx );
	ux /= 2.0f * dx;

	const int pmlIdx = iy * BLOCK_X + threadIdx.x;
	const int gloIdx = iy * lda + ix + HALF_SCHEME;
	
	if(threadIdx.x < NPMLX - HALF_SCHEME) pml2d_gpu_update(pmlIdx, vel[gloIdx], ux, uxx, pml);
}

__global__ static void XRightPML2DUpdateWavefieldKernel
(
	float * __restrict__ out,
	pml2d_gpu_t * pml,
	const int lda, 
	const int nxPad
)
{
	const int thrdBgnX = nxPad - NPMLX;
	const int thrdBgnY = HALF_SCHEME + blockIdx.y * blockDim.y;
	
	const int ix = thrdBgnX + threadIdx.x;
	const int iy = thrdBgnY + threadIdx.y;

	const int pmlIdx = iy * BLOCK_X + threadIdx.x;
	const int gloIdx = iy * lda + ix;
	
	if(threadIdx.x < NPMLX - HALF_SCHEME) out[gloIdx] += pml->d_uxx[pmlIdx];
}

__global__ static void InjectSourceKernel
(
	float * __restrict__ in,
	const float wav,
	const int lda,
	const int sx,
	const int sz
)
{
	if(threadIdx.x == 0 && threadIdx.y == 0) in[sz * lda + sx] += wav;
}

__global__ static void TimeUpdateKernel
(
	float * __restrict__ in,
	const int lda,
	float * __restrict__ out
)
{
	__shared__ float smem[BLOCK_Y][BLOCK_X];

	const int ix = HALF_SCHEME + blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = HALF_SCHEME + blockIdx.y * blockDim.y + threadIdx.y;

	smem[threadIdx.y][threadIdx.x] = - out[iy * lda + ix];
	smem[threadIdx.y][threadIdx.x] += 2.0 * in[iy * lda + ix];

	out[iy * lda + ix] = smem[threadIdx.y][threadIdx.x];
}

__global__ static void SpaceUpdateKernel
(
	float * __restrict__ in,
	float * __restrict__ vel, 
	const int lda,
	const float dx,
	const float dz,
	float * __restrict__ out
)
{
	__shared__ float smem[STENCIL_Y][STENCIL_X];

	const int thrdBgnX = blockIdx.x * blockDim.x;
//	const int thrdEndX = thrdBgnX + blockDim.x + 2 * HALF_SCHEME;
	const int thrdBgnY = blockIdx.y * blockDim.y;
//	const int thrdEndY = thrdBgnY + blockDim.y + 2 * HALF_SCHEME;

//	for(int iy = thrdBgnY + threadIdx.y; iy < thrdEndY; iy += blockDim.y) {
//		for(int ix = thrdBgnX + threadIdx.x; ix < thrdEndX; ix += blockDim.x) {
//			const int idxY = iy - thrdBgnY;
//			const int idxX = ix - thrdBgnX;
//			smem[idxY][idxX] = in[iy * lda + ix];
//		}
//	}
	
	const int iy = thrdBgnY + threadIdx.y;
	const int ix = thrdBgnX + threadIdx.x;
	smem[threadIdx.y][threadIdx.x] = in[iy * lda + ix];
	if(threadIdx.x < 2 * HALF_SCHEME) {
		smem[threadIdx.y][threadIdx.x + blockDim.x] = in[iy * lda + ix + blockDim.x];
	}
	if(threadIdx.y < 2 * HALF_SCHEME) {
		smem[threadIdx.y + blockDim.y][threadIdx.x] = in[(iy + blockDim.y) * lda + ix];
	}
	if(threadIdx.x < 2 * HALF_SCHEME && threadIdx.y < 2 * HALF_SCHEME) {
		smem[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x] = in[(iy + blockDim.y) * lda + ix + blockDim.x];
	}
	
	__syncthreads();

	float derivative_x, derivative_z;
	derivative_x  = fd8_b4 * smem[threadIdx.y + HALF_SCHEME][threadIdx.x + HALF_SCHEME - 4];
	derivative_x += fd8_b3 * smem[threadIdx.y + HALF_SCHEME][threadIdx.x + HALF_SCHEME - 3];
	derivative_x += fd8_b2 * smem[threadIdx.y + HALF_SCHEME][threadIdx.x + HALF_SCHEME - 2];
	derivative_x += fd8_b1 * smem[threadIdx.y + HALF_SCHEME][threadIdx.x + HALF_SCHEME - 1];
	derivative_x += fd8_b0 * smem[threadIdx.y + HALF_SCHEME][threadIdx.x + HALF_SCHEME    ];
	derivative_x += fd8_b1 * smem[threadIdx.y + HALF_SCHEME][threadIdx.x + HALF_SCHEME + 1];
	derivative_x += fd8_b2 * smem[threadIdx.y + HALF_SCHEME][threadIdx.x + HALF_SCHEME + 2];
	derivative_x += fd8_b3 * smem[threadIdx.y + HALF_SCHEME][threadIdx.x + HALF_SCHEME + 3];
	derivative_x += fd8_b4 * smem[threadIdx.y + HALF_SCHEME][threadIdx.x + HALF_SCHEME + 4];

	derivative_x /= (dx * dx);	

	derivative_z  = fd8_b4 * smem[threadIdx.y + HALF_SCHEME - 4][threadIdx.x + HALF_SCHEME];
	derivative_z += fd8_b3 * smem[threadIdx.y + HALF_SCHEME - 3][threadIdx.x + HALF_SCHEME];
	derivative_z += fd8_b2 * smem[threadIdx.y + HALF_SCHEME - 2][threadIdx.x + HALF_SCHEME];
	derivative_z += fd8_b1 * smem[threadIdx.y + HALF_SCHEME - 1][threadIdx.x + HALF_SCHEME];
	derivative_z += fd8_b0 * smem[threadIdx.y + HALF_SCHEME    ][threadIdx.x + HALF_SCHEME];
	derivative_z += fd8_b1 * smem[threadIdx.y + HALF_SCHEME + 1][threadIdx.x + HALF_SCHEME];
	derivative_z += fd8_b2 * smem[threadIdx.y + HALF_SCHEME + 2][threadIdx.x + HALF_SCHEME];
	derivative_z += fd8_b3 * smem[threadIdx.y + HALF_SCHEME + 3][threadIdx.x + HALF_SCHEME];
	derivative_z += fd8_b4 * smem[threadIdx.y + HALF_SCHEME + 4][threadIdx.x + HALF_SCHEME];

	derivative_z /= (dz * dz);

	int idx = (HALF_SCHEME + thrdBgnY + threadIdx.y) * lda + HALF_SCHEME + thrdBgnX + threadIdx.x;
	out[idx] += vel[idx] * (derivative_x + derivative_z);
}

void WflToBin(const char * fileName, const float * wfl, const int length)
{
	FILE * fp;
	if((fp = fopen(const_cast<char*>(fileName), "wb")) == NULL) {
		fprintf(stderr, "cannot open data file!\n");
	} 
	fwrite(wfl, sizeof(float), length, fp);
	fclose(fp);
	return;
}

void SetRickerWavlet(float * wav, const int nt, const float dt, const float f0)
{
	for(int it = 0; it < nt; it++) {
		float ttime = it * dt;
		float temp = M_PI * M_PI * f0 * f0 * (ttime - 1.0 / f0) * (ttime - 1.0 / f0);
		wav[it] = (1.0 - 2.0 * temp) * expf(- temp);
	}	
}

void SetGrid(const int nxx, const int nzz, const int npmlx1, const int npmlx2, const int npmlz1, const int npmlz2, int& nxPad, int& nzPad, int& nx, int& nz, int& x1, int& x2, int& z1, int& z2)
{
	nx = nxx + npmlx1 + npmlx2;
	nz = nzz + npmlz1 + npmlz2;

	nxPad = (nx - 2 * HALF_SCHEME) % BLOCK_X == 0 ? nx - 2 * HALF_SCHEME : ((int)((nx - 2 * HALF_SCHEME) / BLOCK_X) + 1) * BLOCK_X;
	nzPad = (nz - 2 * HALF_SCHEME) % BLOCK_Y == 0 ? nz - 2 * HALF_SCHEME : ((int)((nz - 2 * HALF_SCHEME) / BLOCK_Y) + 1) * BLOCK_Y;
	nxPad += 2 * HALF_SCHEME;
	nzPad += 2 * HALF_SCHEME;

	x1 = (nxPad - nxx) / 2;
	x2 = nxPad - nxx - x1;
	z1 = (nzPad - nzz) / 2;
	z2 = nzPad - nzz - z1;

	return;
}


#define ROUND(x) ((int)((x) + 0.5))

int main(int argc, char * argv[])
{
	PrintDeviceInfo();

	const float T = 1.5;
	const float dx = 0.01;
	const float dz = 0.01;
	const float dt = 0.001;
	const int nxx = 1001;
	const int nzz = 1001;
	const float f0 = 15;
	const int nt = ROUND(T / dt) + 1;
	const float fsx = (nxx - 1) * dx * 0.5;
	const float fsz = (nzz - 1) * dz * 0.5;
	
	int nxPad, nzPad, nx, nz, x1, x2, z1, z2;
	SetGrid(nxx, nzz, NPMLX, NPMLX, NPMLZ, NPMLZ,
		nxPad, nzPad, nx, nz, x1, x2, z1, z2);	

	fprintf(stdout, "INFO: fd grid configuration: \n");
	fprintf(stdout, "INFO: fd grid before padding:                                                    nxx = %d, nzz = %d.\n", nxx, nzz);
	fprintf(stdout, "INFO: fd grid PML padding samples:                                               npmlx1 = %d, npmlx2 = %d, npmlz1 = %d, npmlz2 = %d.\n", NPMLX, NPMLX, NPMLZ, NPMLZ);
	fprintf(stdout, "INFO: fd grid after PML padding:                                                 nx = %d, nz = %d.\n", nx, nz);
	fprintf(stdout, "INFO: fd grid after CUDA padding:                                                nx_pad = %d, nz_pad = %d\n", nxPad, nzPad);
	fprintf(stdout, "INFO: fd grid left padding samples and right padding samples in x direction:    (%d, %d).\n", x1, x2);
	fprintf(stdout, "INFO: fd grid left padding samples and right padding samples in z direction:    (%d, %d).\n", z1, z2);
	
	int sx = x1 + (int)(fsx / dx + 0.5);
	int sz = z1 + (int)(fsz / dz + 0.5); 
	
	fprintf(stdout, "INFO: source configuration: \n");
	fprintf(stdout, "INFO: source location (unit: km),                                                (sx, sz) = (%f, %f).\n", fsx, fsz);
	fprintf(stdout, "INFO: source location (grid points ver GPU),                                     (sz, sz) = (%d, %d).\n", sx, sz);

	float * wav = (float*)malloc(sizeof(float) * nt);
	SetRickerWavlet(wav, nt, dt, f0);	

	int nxz = nxPad * nzPad;

	float * d_vel = NULL;
	float * h_vel = (float*)malloc(sizeof(float) * nxz);
	for(int i = 0; i < nxz; i++) h_vel[i] = 4.0;

	pml2d_gpu_t pmlzt(nxPad, NPMLZ, Z);
	pml2d_gpu_t * d_pmlzt;
	pmlzt.pml2d_gpu_allocate();

	/* PML for z-top */
	for(int iz = 0; iz < NPMLZ; iz++) {
		for(int ix = 0; ix < nxPad; ix++) {
			float L = NPMLZ * dz;
			float d0 = - 3.0f * h_vel[iz * nxPad + ix] * logf(R) / (2.0f * L * L * L);
			float damp_z = d0 * (NPMLZ - iz) * dz * (NPMLZ - iz) * dz;
			float damp_dz = -2.0f * d0 * (NPMLZ - iz) * dz;
			float alpha_z = iz * dz / L * M_PI * f0;
			float alpha_dz = M_PI * f0 / L;
			
			int pml_idx = iz * nxPad + ix;
			pmlzt.pml2d_init(pml_idx, damp_z, damp_dz, alpha_z, alpha_dz, dt);	 
		}
	}	
	pmlzt.pml2d_memcpyh2d();
	cudaMalloc((void**)&d_pmlzt, sizeof(pml2d_gpu_t));
	cudaMemcpy(d_pmlzt, &pmlzt, sizeof(pml2d_gpu_t), cudaMemcpyHostToDevice);

	pml2d_gpu_t pmlzb(nxPad, NPMLZ, Z);
	pml2d_gpu_t * d_pmlzb;
	pmlzb.pml2d_gpu_allocate();

	/* PML for z-top */
	for(int iz = nzPad - NPMLZ; iz < nzPad; iz++) {
		for(int ix = 0; ix < nxPad; ix++) {
			float L = NPMLZ * dz;
			float d0 = - 3.0f * h_vel[iz * nxPad + ix] * logf(R) / (2.0f * L * L * L);
			float damp_z = d0 * (iz - nzPad + 1 + NPMLZ) * dz * (iz - nzPad + 1 + NPMLZ) * dz;
			float damp_dz = 2.0f * d0 * (iz - nzPad + 1 + NPMLZ) * dz;
			float alpha_z = (nzPad - 1 - iz) * dz / L * M_PI * f0;
			float alpha_dz = - M_PI * f0 / L;
			
			int pml_idx = (iz - nzPad + NPMLZ) * nxPad + ix;
			pmlzb.pml2d_init(pml_idx, damp_z, damp_dz, alpha_z, alpha_dz, dt);	 
		}
	}	
	pmlzb.pml2d_memcpyh2d();
	cudaMalloc((void**)&d_pmlzb, sizeof(pml2d_gpu_t));
	cudaMemcpy(d_pmlzb, &pmlzb, sizeof(pml2d_gpu_t), cudaMemcpyHostToDevice);

	pml2d_gpu_t pmlxl(NPMLX, nzPad, X);
	pml2d_gpu_t * d_pmlxl;
	pmlxl.pml2d_gpu_allocate();

	/* PML for x-left */
	for(int iz = 0; iz < nzPad; iz++) {
		for(int ix = 0; ix < NPMLX; ix++) {
			float L = NPMLX * dx;
			float d0 = - 3.0f * h_vel[iz * nxPad + ix] * logf(R) / (2.0f * L * L * L);
			float damp_x = d0 * (NPMLX - ix) * dx * (NPMLX - ix) * dx;
			float damp_dx = -2.0f * d0 * (NPMLX - ix) * dx;
			float alpha_x = ix * dx / L * M_PI * f0;
			float alpha_dx = M_PI * f0 / L;
			
			int pml_idx = iz * BLOCK_X + ix;
			pmlxl.pml2d_init(pml_idx, damp_x, damp_dx, alpha_x, alpha_dx, dt);	 
		}
	}	
	pmlxl.pml2d_memcpyh2d();
	cudaMalloc((void**)&d_pmlxl, sizeof(pml2d_gpu_t));
	cudaMemcpy(d_pmlxl, &pmlxl, sizeof(pml2d_gpu_t), cudaMemcpyHostToDevice);

	pml2d_gpu_t pmlxr(NPMLX, nzPad, X);
	pml2d_gpu_t * d_pmlxr;
	pmlxr.pml2d_gpu_allocate();

	/* PML for x-right */
	for(int iz = 0; iz < nzPad; iz++) {
		for(int ix = nxPad - NPMLX; ix < nxPad; ix++) {
			float L = NPMLX * dx;
			float d0 = - 3.0f * h_vel[iz * nxPad + ix] * logf(R) / (2.0f * L * L * L);
			float damp_x = d0 * (ix - nxPad + 1 + NPMLX) * dx * (ix - nxPad + 1 + NPMLX) * dx;
			float damp_dx = 2.0f * d0 * (ix - nxPad + 1 + NPMLX) * dx;
			float alpha_x = (nxPad - 1 - ix) * dx / L * M_PI * f0;
			float alpha_dx = - M_PI * f0 / L;
			
			int pml_idx = iz * BLOCK_X + ix - nxPad + NPMLX;
			pmlxr.pml2d_init(pml_idx, damp_x, damp_dx, alpha_x, alpha_dx, dt);	 
		}
	}	
	pmlxr.pml2d_memcpyh2d();
	cudaMalloc((void**)&d_pmlxr, sizeof(pml2d_gpu_t));
	cudaMemcpy(d_pmlxr, &pmlxr, sizeof(pml2d_gpu_t), cudaMemcpyHostToDevice);

	for(int i = 0; i < nxz; i++) h_vel[i] = h_vel[i] * h_vel[i] * dt * dt;
	cudaMalloc((void**)&d_vel, sizeof(float) * nxz);
	cudaMemcpy(d_vel, h_vel, sizeof(float) * nxz, cudaMemcpyHostToDevice);

	float * h_uo = (float*)malloc(sizeof(float) * nxz);
	float * d_uo;
	float * d_um;
	cudaMalloc((void**)&d_uo, sizeof(float) * nxz);
	cudaMalloc((void**)&d_um, sizeof(float) * nxz);
	cudaMemset(d_uo, 0, sizeof(float) * nxz);
	cudaMemset(d_um, 0, sizeof(float) * nxz);

	int gridX = (nxPad - 2 * HALF_SCHEME) / BLOCK_X;
	int gridY = (nzPad - 2 * HALF_SCHEME) / BLOCK_Y;
	
	dim3 grid( gridX, gridY, 1 );
	dim3 threads( BLOCK_X, BLOCK_Y, 1 );

	dim3 grid_z( gridX, 1, 1);
	dim3 threads_z( BLOCK_X, BLOCK_PML_Y, 1);

	assert(BLOCK_X > NPMLX);
	dim3 grid_x( 1, gridY, 1);
	dim3 threads_x( BLOCK_X, BLOCK_Y, 1);

	float gpuTime = 0.0;
	
	cudaEvent_t start;
	cudaEvent_t finish;

	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start,0);

	for(int it = 0; it < nt; it++) {
		InjectSourceKernel<<<1, threads>>>(d_uo, wav[it], nxPad, sx, sz);

		ZTopPML2DUpdateKernel<<<grid_z, threads_z>>>(d_um, d_uo, d_vel, nxPad, dx, dz, d_pmlzt);

		ZBottomPML2DUpdateKernel<<<grid_z, threads_z>>>(d_um, d_uo, d_vel, nxPad, dx, dz, nzPad, d_pmlzb);
	
		XLeftPML2DUpdateKernel<<<grid_x, threads_x>>>(d_um, d_uo, d_vel, nxPad, dx, dz, d_pmlxl);
	
		XRightPML2DUpdateKernel<<<grid_x, threads_x>>>(d_um, d_uo, d_vel, nxPad, dx, dz, nxPad, d_pmlxr);

		TimeUpdateKernel<<<grid, threads>>>(d_uo, nxPad, d_um);

		SpaceUpdateKernel<<<grid, threads>>>(d_uo, d_vel, nxPad, dx, dz, d_um);
		
		ZTopPML2DUpdateWavefieldKernel<<<grid_z, threads_z>>>(d_um, d_pmlzt, nxPad);

		ZBottomPML2DUpdateWavefieldKernel<<<grid_z, threads_z>>>(d_um, d_pmlzb, nxPad, nzPad);

		XLeftPML2DUpdateWavefieldKernel<<<grid_x, threads_x>>>(d_um, d_pmlxl, nxPad);

		XRightPML2DUpdateWavefieldKernel<<<grid_x, threads_x>>>(d_um, d_pmlxr, nxPad, nxPad);

		cudaDeviceSynchronize();

		{
			float * swapPtr = d_uo;
			d_uo = d_um;
			d_um = swapPtr;
		}
	}

	cudaEventRecord(finish,0);

	cudaEventSynchronize(finish);
	
	cudaEventElapsedTime(&gpuTime,start,finish);

	cudaEventDestroy(start);
	cudaEventDestroy(finish);

	fprintf(stderr, "elapsed time of GPU fd is %f s\n", gpuTime * 0.001);

	cudaMemcpy(h_uo, d_uo, sizeof(float) * nxz, cudaMemcpyDeviceToHost);

	WflToBin("./wfl_gpu.dat", h_uo, nxz);

	cudaFree(d_uo);
	cudaFree(d_um);
	cudaFree(d_vel);

	cudaFree(d_pmlzt);
	cudaFree(d_pmlzb);
	cudaFree(d_pmlxl);

	free(h_uo); h_uo = NULL;
	free(h_vel); h_vel = NULL;
		
	const int threadNums = 8;
	omp_set_num_threads(threadNums);

	sx = NPMLX + (int)(fsx / dx + 0.5);
	sz = NPMLZ + (int)(fsz / dz + 0.5);

	fprintf(stdout, "INFO: source configuration: \n");
	fprintf(stdout, "INFO: source location (unit: km),                                                (sx, sz) = (%f, %f).\n", fsx, fsz);
	fprintf(stdout, "INFO: source location (grid points ver CPU),                                     (sz, sz) = (%d, %d).\n", sx, sz);

	float * h_um = (float*)malloc(sizeof(float) * nx * nz);
	h_uo = (float*)malloc(sizeof(float) * nx * nz);
	h_vel = (float*)malloc(sizeof(float) * nx * nz);
	for(int i = 0; i < nx * nz; i++) h_vel[i] = 4.0;
	for(int i = 0; i < nx * nz; i++) h_vel[i] = h_vel[i] * h_vel[i] * dt * dt;

	memset(h_um, 0, sizeof(float) * nx * nz);
	memset(h_uo, 0, sizeof(float) * nx * nz);
	double t0 = omp_get_wtime();
	
	for(int it = 0; it < nt; it++) {
		h_uo[sz * nx + sx] += wav[it];

		#pragma omp parallel for num_threads(threadNums) schedule(static, 1) shared(h_uo, h_um)
		for(int iz = HALF_SCHEME; iz < nz - HALF_SCHEME; iz++) {
			for(int ix = HALF_SCHEME; ix < nx - HALF_SCHEME; ix++) {
				int gloIdx = iz * nx + ix;
				float uxx = fd8_b0 * h_uo[gloIdx] + fd8_b1 * (h_uo[gloIdx + 1] + h_uo[gloIdx - 1]) + fd8_b2 * (h_uo[gloIdx + 2] + h_uo[gloIdx - 2]) + fd8_b3 * (h_uo[gloIdx + 3] + h_uo[gloIdx - 3]) + fd8_b4 * (h_uo[gloIdx + 4] + h_uo[gloIdx - 4]);
				float uzz = fd8_b0 * h_uo[gloIdx] + fd8_b1 * (h_uo[gloIdx + 1 * nx] + h_uo[gloIdx - 1 * nx]) + fd8_b2 * (h_uo[gloIdx + 2 * nx] + h_uo[gloIdx - 2 * nx]) + fd8_b3 * (h_uo[gloIdx + 3 * nx] + h_uo[gloIdx - 3 * nx]) + fd8_b4 * (h_uo[gloIdx + 4 * nx] + h_uo[gloIdx - 4 * nx]);
				uxx /= (dx * dx);
				uzz /= (dz * dz);
				h_um[gloIdx] = -h_um[gloIdx] + 2.0 * h_uo[gloIdx] + h_vel[gloIdx] * (uxx + uzz);
			}
		}
		{
			float * swapPtr = h_uo;
			h_uo = h_um;
			h_um = swapPtr;
		}
	}
	double t1 = omp_get_wtime() - t0;

	WflToBin("./wfl_cpu.dat", h_uo, nx * nz);

	fprintf(stderr, "elapsed time of CPU fd is %f s\n", t1);

	free(h_uo);
	free(h_um);
	free(h_vel);
	free(wav);

	return 0;
}
