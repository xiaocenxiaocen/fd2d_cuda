#include <cstdio>
#include <math.h>
#include <omp.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define fd8_b0 -2.847222222f
#define fd8_b1 1.6f
#define fd8_b2 -0.2f
#define fd8_b3 0.02539682540f
#define fd8_b4 -0.001785714286f

#define fd8_a1 0.8f
#define fd8_a2 -0.2f
#define fd8_a3 0.038095238f
#define fd8_a4 -0.0035714286f

#define SCHEME 8
#define HALF_SCHEME (SCHEME / 2)
#define BLOCK_X 32
#define BLOCK_Y 10
#define STENCIL_X BLOCK_X + 2 * HALF_SCHEME
#define STENCIL_Y BLOCK_Y + 2 * HALF_SCHEME

#define ALIGNED_32 (128 / sizeof(float))

#define R 1e-8

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
#define NPMLZ 10
#define NPMLX 10

__global__ static void fdtd2d_kernel(float * __restrict__ uo, float * __restrict__ vel, const int lda, const float invsqrdx, const float invsqrdz, float * __restrict__ um, pml2d_gpu_t * pmlxl, pml2d_gpu_t * pmlxr, pml2d_gpu_t * pmlzt, pml2d_gpu_t * pmlzb)
{
	__shared__ float smem[STENCIL_Y][STENCIL_X];
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int ltidx = tx + HALF_SCHEME;
	const int ltidy = ty + HALF_SCHEME;
	const int gtidx = blockIdx.x * blockDim.x + tx + ALIGNED_32;
	const int gtidy = blockIdx.y * blockDim.y + ty + HALF_SCHEME;
	const int gidx = gtidy * lda + gtidx;
	smem[ltidy][ltidx] = uo[gidx];
	// top & bottom
	if(ty < HALF_SCHEME) {
		smem[ty ][ltidx] = uo[gidx - HALF_SCHEME * lda];
		smem[ty + BLOCK_Y + HALF_SCHEME][ltidx] = uo[gidx + BLOCK_Y * lda];
	}
	// left & right
	if(tx < HALF_SCHEME) {
		smem[ltidy][tx ] = uo[gidx - HALF_SCHEME];
		smem[ltidy][tx + BLOCK_X + HALF_SCHEME] = uo[gidx + BLOCK_X];
	}
	__syncthreads();
	float uxx, uzz;
	uxx = fd8_b0 * smem[ltidy][ltidx ]
	    + fd8_b1 * (smem[ltidy][ltidx + 1] + smem[ltidy][ltidx - 1])
	    + fd8_b2 * (smem[ltidy][ltidx + 2] + smem[ltidy][ltidx - 2])
	    + fd8_b3 * (smem[ltidy][ltidx + 3] + smem[ltidy][ltidx - 3])
	    + fd8_b4 * (smem[ltidy][ltidx + 4] + smem[ltidy][ltidx - 4]);
	uzz = fd8_b0 * smem[ltidy ][ltidx]
	    + fd8_b1 * (smem[ltidy + 1][ltidx] + smem[ltidy - 1][ltidx])
	    + fd8_b2 * (smem[ltidy + 2][ltidx] + smem[ltidy - 2][ltidx])
	    + fd8_b3 * (smem[ltidy + 3][ltidx] + smem[ltidy - 3][ltidx])
	    + fd8_b4 * (smem[ltidy + 4][ltidx] + smem[ltidy - 4][ltidx]);
	uxx *= invsqrdx;
	uzz *= invsqrdz;

	um[gidx] = -um[gidx] + 2.0 * uo[gidx] + vel[gidx] * (uxx + uzz);
	if(blockIdx.x == 0) {
		const int pmlidx = gtidy * BLOCK_X + tx;
		if(tx < NPMLX) {
			um[gidx] += pmlxl->d_uxx[pmlidx];
		}	
	}
	if(blockIdx.x == gridDim.x - 1) {
		const int pmlidx = gtidy * BLOCK_X + tx;
		if(tx >= blockDim.x - NPMLX) {
			um[gidx] += pmlxr->d_uxx[pmlidx];
		}	
	}
	if(blockIdx.y == 0) {
		const int pmlidx = ty * lda + gtidx;
		if(ty < NPMLZ) {
			um[gidx] += pmlzt->d_uxx[pmlidx];
		}	
	}
	if(blockIdx.y == gridDim.y - 1) {
		const int pmlidx = (ty - blockDim.y + NPMLZ) * lda + gtidx;
		if(ty >= blockDim.y - NPMLZ) {
			um[gidx] += pmlzb->d_uxx[pmlidx];
		}	
	}
}

__global__ static void pml2d_kernel(float * __restrict__ um, float * __restrict__ uo, float * __restrict__ vel, const int lda, const float dx, const float dz, pml2d_gpu_t * pmlxl, pml2d_gpu_t * pmlxr, pml2d_gpu_t * pmlzt, pml2d_gpu_t * pmlzb)
{
	__shared__ float smem[STENCIL_Y][STENCIL_X];
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int ltidx = tx + HALF_SCHEME;
	const int ltidy = ty + HALF_SCHEME;
	const int gtidx = blockIdx.x * blockDim.x + tx + ALIGNED_32;
	const int gtidy = blockIdx.y * blockDim.y + ty + HALF_SCHEME;
	const int gidx = gtidy * lda + gtidx;
	float ux = 0.0;
	float uz = 0.0;
	float uxx = 0.0;
	float uzz = 0.0;
	if(blockIdx.x == 0) {
		smem[ltidy][ltidx] = um[gidx];
		if(tx < HALF_SCHEME) {
			smem[ltidy][tx] = um[gidx - HALF_SCHEME];
		}
		__syncthreads();
		if(tx < NPMLX) {
			ux = fd8_a1 * (smem[ltidy][ltidx + 1] - smem[ltidy][ltidx - 1])
			   + fd8_a2 * (smem[ltidy][ltidx + 2] - smem[ltidy][ltidx - 2])
			   + fd8_a3 * (smem[ltidy][ltidx + 3] - smem[ltidy][ltidx - 3])
			   + fd8_a4 * (smem[ltidy][ltidx + 4] - smem[ltidy][ltidx - 4]);
			uxx = fd8_b0 * smem[ltidy][ltidx ]
			    + fd8_b1 * (smem[ltidy][ltidx + 1] + smem[ltidy][ltidx - 1])
			    + fd8_b2 * (smem[ltidy][ltidx + 2] + smem[ltidy][ltidx - 2])
			    + fd8_b3 * (smem[ltidy][ltidx + 3] + smem[ltidy][ltidx - 3])
			    + fd8_b4 * (smem[ltidy][ltidx + 4] + smem[ltidy][ltidx - 4]);
		}
		__syncthreads();
		smem[ltidy][ltidx] = uo[gidx];
		if(tx < HALF_SCHEME) {
			smem[ltidy][tx] = uo[gidx - HALF_SCHEME];
		}
		__syncthreads();
		if(tx < NPMLX) {
			ux += fd8_a1 * (smem[ltidy][ltidx + 1] - smem[ltidy][ltidx - 1])
			   + fd8_a2 * (smem[ltidy][ltidx + 2] - smem[ltidy][ltidx - 2])
			   + fd8_a3 * (smem[ltidy][ltidx + 3] - smem[ltidy][ltidx - 3])
			   + fd8_a4 * (smem[ltidy][ltidx + 4] - smem[ltidy][ltidx - 4]);
			uxx += fd8_b0 * smem[ltidy][ltidx ]
			    + fd8_b1 * (smem[ltidy][ltidx + 1] + smem[ltidy][ltidx - 1])
			    + fd8_b2 * (smem[ltidy][ltidx + 2] + smem[ltidy][ltidx - 2])
			    + fd8_b3 * (smem[ltidy][ltidx + 3] + smem[ltidy][ltidx - 3])
			    + fd8_b4 * (smem[ltidy][ltidx + 4] + smem[ltidy][ltidx - 4]);
		}
		__syncthreads();
		ux /= 2.0f * dx;
		uxx /= 2.0f * dx * dx;
		// TO CONFIRM
		const int pmlidx = gtidy * BLOCK_X + tx;
		if(tx < NPMLX) {
			pml2d_gpu_update(pmlidx, vel[gidx], ux, uxx, pmlxl);
		}
	}
	if(blockIdx.x == gridDim.x - 1) {
		smem[ltidy][ltidx] = um[gidx];
		if(tx < HALF_SCHEME) {
			smem[ltidy][ltidx + BLOCK_X] = um[gidx + BLOCK_X];
		}
		__syncthreads();
		if(tx >= blockDim.x - NPMLX) {
			ux = fd8_a1 * (smem[ltidy][ltidx + 1] - smem[ltidy][ltidx - 1])
			   + fd8_a2 * (smem[ltidy][ltidx + 2] - smem[ltidy][ltidx - 2])
			   + fd8_a3 * (smem[ltidy][ltidx + 3] - smem[ltidy][ltidx - 3])
			   + fd8_a4 * (smem[ltidy][ltidx + 4] - smem[ltidy][ltidx - 4]);
			uxx = fd8_b0 * smem[ltidy][ltidx ]
			    + fd8_b1 * (smem[ltidy][ltidx + 1] + smem[ltidy][ltidx - 1])
			    + fd8_b2 * (smem[ltidy][ltidx + 2] + smem[ltidy][ltidx - 2])
			    + fd8_b3 * (smem[ltidy][ltidx + 3] + smem[ltidy][ltidx - 3])
			    + fd8_b4 * (smem[ltidy][ltidx + 4] + smem[ltidy][ltidx - 4]);
		}
		__syncthreads();
		smem[ltidy][ltidx] = uo[gidx];
		if(tx < HALF_SCHEME) {
			smem[ltidy][ltidx + BLOCK_X] = uo[gidx + BLOCK_X];
		}
		__syncthreads();
		if(tx >= blockDim.x - NPMLX) {
			ux += fd8_a1 * (smem[ltidy][ltidx + 1] - smem[ltidy][ltidx - 1])
			   + fd8_a2 * (smem[ltidy][ltidx + 2] - smem[ltidy][ltidx - 2])
			   + fd8_a3 * (smem[ltidy][ltidx + 3] - smem[ltidy][ltidx - 3])
			   + fd8_a4 * (smem[ltidy][ltidx + 4] - smem[ltidy][ltidx - 4]);
			uxx += fd8_b0 * smem[ltidy][ltidx ]
			    + fd8_b1 * (smem[ltidy][ltidx + 1] + smem[ltidy][ltidx - 1])
			    + fd8_b2 * (smem[ltidy][ltidx + 2] + smem[ltidy][ltidx - 2])
			    + fd8_b3 * (smem[ltidy][ltidx + 3] + smem[ltidy][ltidx - 3])
			    + fd8_b4 * (smem[ltidy][ltidx + 4] + smem[ltidy][ltidx - 4]);
		}
		__syncthreads();
		ux /= 2.0f * dx;
		uxx /= 2.0f * dx * dx;
		// TO CONFIRM
		const int pmlidx = gtidy * BLOCK_X + tx;
		if(tx >= blockDim.x - NPMLX) {
			pml2d_gpu_update(pmlidx, vel[gidx], ux, uxx, pmlxr);
		}
	}
	__syncthreads();
	if(blockIdx.y == 0) {
		smem[ltidy][ltidx] = um[gidx];
		if(ty < HALF_SCHEME) {
			smem[ty][ltidx] = um[gidx - HALF_SCHEME * lda];
			smem[ty + BLOCK_Y + HALF_SCHEME][ltidx] = um[gidx + BLOCK_Y * lda];
		}
		__syncthreads();
		if(ty < NPMLZ) {
			uz = fd8_a1 * (smem[ltidy + 1][ltidx] - smem[ltidy - 1][ltidx])
			  + fd8_a2 * (smem[ltidy + 2][ltidx] - smem[ltidy - 2][ltidx])
			  + fd8_a3 * (smem[ltidy + 3][ltidx] - smem[ltidy - 3][ltidx])
			  + fd8_a4 * (smem[ltidy + 4][ltidx] - smem[ltidy - 4][ltidx]);
			uzz = fd8_b0 * smem[ltidy ][ltidx]
			    + fd8_b1 * (smem[ltidy + 1][ltidx] + smem[ltidy - 1][ltidx])
			    + fd8_b2 * (smem[ltidy + 2][ltidx] + smem[ltidy - 2][ltidx])
			    + fd8_b3 * (smem[ltidy + 3][ltidx] + smem[ltidy - 3][ltidx])
			    + fd8_b4 * (smem[ltidy + 4][ltidx] + smem[ltidy - 4][ltidx]);
		}
		__syncthreads();
		smem[ltidy][ltidx] = uo[gidx];
		if(ty < HALF_SCHEME) {
			smem[ty][ltidx] = uo[gidx - HALF_SCHEME * lda];
			smem[ty + BLOCK_Y + HALF_SCHEME][ltidx] = uo[gidx + BLOCK_Y * lda];
		}
		__syncthreads();
		if(ty < NPMLZ) {
			uz += fd8_a1 * (smem[ltidy + 1][ltidx] - smem[ltidy - 1][ltidx])
			   + fd8_a2 * (smem[ltidy + 2][ltidx] - smem[ltidy - 2][ltidx])
			   + fd8_a3 * (smem[ltidy + 3][ltidx] - smem[ltidy - 3][ltidx])
			   + fd8_a4 * (smem[ltidy + 4][ltidx] - smem[ltidy - 4][ltidx]);
			uzz += fd8_b0 * smem[ltidy ][ltidx]
			    + fd8_b1 * (smem[ltidy + 1][ltidx] + smem[ltidy - 1][ltidx])
			    + fd8_b2 * (smem[ltidy + 2][ltidx] + smem[ltidy - 2][ltidx])
			    + fd8_b3 * (smem[ltidy + 3][ltidx] + smem[ltidy - 3][ltidx])
			    + fd8_b4 * (smem[ltidy + 4][ltidx] + smem[ltidy - 4][ltidx]);
		}
		__syncthreads();
		uz /= 2.0f * dz;
		uzz /= 2.0f * dz * dz;
		// TO CONFIRM
		const int pmlidx = ty * lda + gtidx;
		if(ty < NPMLZ) {
			pml2d_gpu_update(pmlidx, vel[gidx], uz, uzz, pmlzt);
		}
	}
	if(blockIdx.y == gridDim.y - 1) {
		smem[ltidy][ltidx] = um[gidx];
		if(ty < HALF_SCHEME) {
			smem[ty][ltidx] = um[gidx - HALF_SCHEME * lda];
			smem[ty + BLOCK_Y + HALF_SCHEME][ltidx] = um[gidx + BLOCK_Y * lda];
		}
		__syncthreads();
		if(ty >= blockDim.y - NPMLZ) {
			uz = fd8_a1 * (smem[ltidy + 1][ltidx] - smem[ltidy - 1][ltidx])
			   + fd8_a2 * (smem[ltidy + 2][ltidx] - smem[ltidy - 2][ltidx])
			   + fd8_a3 * (smem[ltidy + 3][ltidx] - smem[ltidy - 3][ltidx])
			   + fd8_a4 * (smem[ltidy + 4][ltidx] - smem[ltidy - 4][ltidx]);
			uzz = fd8_b0 * smem[ltidy ][ltidx]
			    + fd8_b1 * (smem[ltidy + 1][ltidx] + smem[ltidy - 1][ltidx])
			    + fd8_b2 * (smem[ltidy + 2][ltidx] + smem[ltidy - 2][ltidx])
			    + fd8_b3 * (smem[ltidy + 3][ltidx] + smem[ltidy - 3][ltidx])
			    + fd8_b4 * (smem[ltidy + 4][ltidx] + smem[ltidy - 4][ltidx]);
		}
		__syncthreads();
		smem[ltidy][ltidx] = uo[gidx];
		if(ty < HALF_SCHEME) {
			smem[ty][ltidx] = uo[gidx - HALF_SCHEME * lda];
			smem[ty + BLOCK_Y + HALF_SCHEME][ltidx] = uo[gidx + BLOCK_Y * lda];
		}
		__syncthreads();
		if(ty >= blockDim.y - NPMLZ) {
			uz += fd8_a1 * (smem[ltidy + 1][ltidx] - smem[ltidy - 1][ltidx])
			   + fd8_a2 * (smem[ltidy + 2][ltidx] - smem[ltidy - 2][ltidx])
			   + fd8_a3 * (smem[ltidy + 3][ltidx] - smem[ltidy - 3][ltidx])
			   + fd8_a4 * (smem[ltidy + 4][ltidx] - smem[ltidy - 4][ltidx]);
			uzz += fd8_b0 * smem[ltidy ][ltidx]
			    + fd8_b1 * (smem[ltidy + 1][ltidx] + smem[ltidy - 1][ltidx])
			    + fd8_b2 * (smem[ltidy + 2][ltidx] + smem[ltidy - 2][ltidx])
			    + fd8_b3 * (smem[ltidy + 3][ltidx] + smem[ltidy - 3][ltidx])
			    + fd8_b4 * (smem[ltidy + 4][ltidx] + smem[ltidy - 4][ltidx]);
		}
		__syncthreads();
		uz /= 2.0f * dz;
		uzz /= 2.0f * dz * dz;
		// TO CONFIRM
		const int pmlidx = (ty - blockDim.y + NPMLZ) * lda + gtidx;
		if(ty >= blockDim.y - NPMLZ) {
			pml2d_gpu_update(pmlidx, vel[gidx], uz, uzz, pmlzb);
		}
	}
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

	nxPad = nx % BLOCK_X == 0 ? nx : ((int)(nx / BLOCK_X) + 1) * BLOCK_X;
	nzPad = nz % BLOCK_Y == 0 ? nz : ((int)(nz / BLOCK_Y) + 1) * BLOCK_Y;
	nxPad += 2 * ALIGNED_32;
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

	const float T = 1.8;
	const float dx = 0.01;
	const float dz = 0.01;
	const float dt = 0.001;
	const int nxx = 1032;
	const int nzz = 1032;
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
	for(int iz = HALF_SCHEME; iz < NPMLZ + HALF_SCHEME; iz++) {
		for(int ix = 0; ix < nxPad; ix++) {
			float L = NPMLZ * dz;
			float d0 = - 3.0f * h_vel[iz * nxPad + ix] * logf(R) / (2.0f * L * L * L);
			float damp_z = d0 * (NPMLZ - iz + HALF_SCHEME) * dz * (NPMLZ - iz + HALF_SCHEME) * dz;
			float damp_dz = -2.0f * d0 * (NPMLZ - iz + HALF_SCHEME) * dz;
			float alpha_z = (iz - HALF_SCHEME) * dz / L * M_PI * f0;
			float alpha_dz = M_PI * f0 / L;
			
			int pml_idx = (iz - HALF_SCHEME) * nxPad + ix;
			pmlzt.pml2d_init(pml_idx, damp_z, damp_dz, alpha_z, alpha_dz, dt);	 
		}
	}	
	pmlzt.pml2d_memcpyh2d();
	cudaMalloc((void**)&d_pmlzt, sizeof(pml2d_gpu_t));
	cudaMemcpy(d_pmlzt, &pmlzt, sizeof(pml2d_gpu_t), cudaMemcpyHostToDevice);

	pml2d_gpu_t pmlzb(nxPad, NPMLZ, Z);
	pml2d_gpu_t * d_pmlzb;
	pmlzb.pml2d_gpu_allocate();

	/* PML for z-bottom */
	for(int iz = nzPad - HALF_SCHEME - NPMLZ; iz < nzPad - HALF_SCHEME; iz++) {
		for(int ix = 0; ix < nxPad; ix++) {
			float L = NPMLZ * dz;
			float d0 = - 3.0f * h_vel[iz * nxPad + ix] * logf(R) / (2.0f * L * L * L);
			float damp_z = d0 * (iz - nzPad + 1 + NPMLZ + HALF_SCHEME) * dz * (iz - nzPad + 1 + NPMLZ + HALF_SCHEME) * dz;
			float damp_dz = 2.0f * d0 * (iz - nzPad + 1 + NPMLZ + HALF_SCHEME) * dz;
			float alpha_z = (nzPad - 1 - iz - HALF_SCHEME) * dz / L * M_PI * f0;
			float alpha_dz = - M_PI * f0 / L;
			
			int pml_idx = (iz - nzPad + NPMLZ + HALF_SCHEME) * nxPad + ix;
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
		for(int ix = ALIGNED_32; ix < ALIGNED_32 + NPMLX; ix++) {
			float L = NPMLX * dx;
			float d0 = - 3.0f * h_vel[iz * nxPad + ix] * logf(R) / (2.0f * L * L * L);
			float damp_x = d0 * (NPMLX + ALIGNED_32 - ix) * dx * (NPMLX + ALIGNED_32 - ix) * dx;
			float damp_dx = -2.0f * d0 * (NPMLX + ALIGNED_32 - ix) * dx;
			float alpha_x = (ix - ALIGNED_32) * dx / L * M_PI * f0;
			float alpha_dx = M_PI * f0 / L;
			
			int pml_idx = iz * BLOCK_X + ix - ALIGNED_32;
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
		for(int ix = nxPad - ALIGNED_32 - NPMLX; ix < nxPad - ALIGNED_32; ix++) {
			float L = NPMLX * dx;
			float d0 = - 3.0f * h_vel[iz * nxPad + ix] * logf(R) / (2.0f * L * L * L);
			float damp_x = d0 * (ix - nxPad + 1 + NPMLX + ALIGNED_32) * dx * (ix - nxPad + 1 + NPMLX + ALIGNED_32) * dx;
			float damp_dx = 2.0f * d0 * (ix - nxPad + 1 + NPMLX + ALIGNED_32) * dx;
			float alpha_x = (nxPad - 1 - ix - ALIGNED_32) * dx / L * M_PI * f0;
			float alpha_dx = - M_PI * f0 / L;
			
			int pml_idx = iz * BLOCK_X + ix - nxPad + ALIGNED_32 + BLOCK_X;
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

	int gridX = (nxPad - 2 * ALIGNED_32) / BLOCK_X;
	int gridY = (nzPad - 2 * HALF_SCHEME) / BLOCK_Y;
	
	dim3 grid( gridX, gridY, 1 );
	dim3 threads( BLOCK_X, BLOCK_Y, 1 );

	float gpuTime = 0.0;
	
	cudaEvent_t start;
	cudaEvent_t finish;

	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start,0);

	float invsqrdx = 1.0 / (dx * dx);
	float invsqrdz = 1.0 / (dz * dz);

	for(int it = 0; it < nt; it++) {
		InjectSourceKernel<<<1, threads>>>(d_uo, wav[it], nxPad, sx, sz);

		pml2d_kernel<<<grid, threads>>>(d_um, d_uo, d_vel, nxPad, dx, dz, d_pmlxl, d_pmlxr, d_pmlzt, d_pmlzb);
		
		fdtd2d_kernel<<<grid, threads>>>(d_uo, d_vel, nxPad, invsqrdx, invsqrdz, d_um, d_pmlxl, d_pmlxr, d_pmlzt, d_pmlzb);

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
	fprintf(stderr, "%f gflop/s\n", (11.0 * nxz * nt / (1024 * 1024 * 1024.0 * gpuTime * 0.001)));

	cudaMemcpy(h_uo, d_uo, sizeof(float) * nxz, cudaMemcpyDeviceToHost);

	WflToBin("./wfl_gpu.dat", h_uo, nxz);

	cudaFree(d_uo);
	cudaFree(d_um);
	cudaFree(d_vel);

	cudaFree(d_pmlzt);
	cudaFree(d_pmlzb);
	cudaFree(d_pmlxl);
	cudaFree(d_pmlxr);

	free(h_uo); h_uo = NULL;
	free(h_vel); h_vel = NULL;
		
	return 0;
}
