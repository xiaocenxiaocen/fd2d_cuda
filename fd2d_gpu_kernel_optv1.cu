__global__ static void fdtd2d_kernel(float * __restrict__ uo, float * __restrict__ vel, const int lda, const float invsqrdx, const float invsqrdz, float * __restrict__ um)
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
		smem[ty                        ][ltidx] = uo[gidx - HALF_SCHEME * lda];
		smem[ty + BLOCK_Y + HALF_SCHEME][ltidx] = uo[gidx + BLOCK_Y * lda];
	}
	// left & right
	if(tx < HALF_SCHEME) {
		smem[ltidy][tx                        ] = uo[gidx - HALF_SCHEME];
		smem[ltidy][tx + BLOCK_X + HALF_SCHEME] = uo[gidx + BLOCK_X];
	}
	__syncthreads();

	float uxx, uzz;
	uxx = fd8_b0 *  smem[ltidy][ltidx    ]  
	    + fd8_b1 * (smem[ltidy][ltidx + 1] + smem[ltidy][ltidx - 1])
	    + fd8_b2 * (smem[ltidy][ltidx + 2] + smem[ltidy][ltidx - 2])
	    + fd8_b3 * (smem[ltidy][ltidx + 3] + smem[ltidy][ltidx - 3])
	    + fd8_b4 * (smem[ltidy][ltidx + 4] + smem[ltidy][ltidx - 4]);
	
	uzz = fd8_b0 *  smem[ltidy    ][ltidx] 
	    + fd8_b1 * (smem[ltidy + 1][ltidx] + smem[ltidy - 1][ltidx])
	    + fd8_b2 * (smem[ltidy + 2][ltidx] + smem[ltidy - 2][ltidx])
	    + fd8_b3 * (smem[ltidy + 3][ltidx] + smem[ltidy - 3][ltidx])
	    + fd8_b4 * (smem[ltidy + 4][ltidx] + smem[ltidy - 4][ltidx]);

	um[gidx] = -um[gidx] + 2.0 * uo[gidx] + vel[gidx] * (uxx + uzz);
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
			   + fd8_a3 * (smem[ltidy][ltidx + 3] - smem[ltidy][ltidx - 2])
			   + fd8_a4 * (smem[ltidy][ltidx + 4] - smem[ltidy][ltidx - 4]);		
			uxx = fd8_b0 *  smem[ltidy][ltidx    ]  
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
			   +  fd8_a2 * (smem[ltidy][ltidx + 2] - smem[ltidy][ltidx - 2])
			   +  fd8_a3 * (smem[ltidy][ltidx + 3] - smem[ltidy][ltidx - 2])
			   +  fd8_a4 * (smem[ltidy][ltidx + 4] - smem[ltidy][ltidx - 4]);		
			uxx += fd8_b0 *  smem[ltidy][ltidx    ]  
			    +  fd8_b1 * (smem[ltidy][ltidx + 1] + smem[ltidy][ltidx - 1])
			    +  fd8_b2 * (smem[ltidy][ltidx + 2] + smem[ltidy][ltidx - 2])
			    +  fd8_b3 * (smem[ltidy][ltidx + 3] + smem[ltidy][ltidx - 3])
			    +  fd8_b4 * (smem[ltidy][ltidx + 4] + smem[ltidy][ltidx - 4]); 
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
			smem[ltidy][tx] = um[gidx + BLOCK_X];
		}

		__syncthreads();
		
		if(tx >= blockDim.x - NPMLX) {
			ux = fd8_a1 * (smem[ltidy][ltidx + 1] - smem[ltidy][ltidx - 1])
			   + fd8_a2 * (smem[ltidy][ltidx + 2] - smem[ltidy][ltidx - 2])
			   + fd8_a3 * (smem[ltidy][ltidx + 3] - smem[ltidy][ltidx - 2])
			   + fd8_a4 * (smem[ltidy][ltidx + 4] - smem[ltidy][ltidx - 4]);		
			uxx = fd8_b0 *  smem[ltidy][ltidx    ]  
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
			   +  fd8_a2 * (smem[ltidy][ltidx + 2] - smem[ltidy][ltidx - 2])
			   +  fd8_a3 * (smem[ltidy][ltidx + 3] - smem[ltidy][ltidx - 2])
			   +  fd8_a4 * (smem[ltidy][ltidx + 4] - smem[ltidy][ltidx - 4]);		
			uxx += fd8_b0 *  smem[ltidy][ltidx    ]  
			    +  fd8_b1 * (smem[ltidy][ltidx + 1] + smem[ltidy][ltidx - 1])
			    +  fd8_b2 * (smem[ltidy][ltidx + 2] + smem[ltidy][ltidx - 2])
			    +  fd8_b3 * (smem[ltidy][ltidx + 3] + smem[ltidy][ltidx - 3])
			    +  fd8_b4 * (smem[ltidy][ltidx + 4] + smem[ltidy][ltidx - 4]); 
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

	if(blockIdx.y == 0) {
		smem[ltidy][ltidx] = um[gidx];
		if(ty < HALF_SCHEME) {
			smem[ty][ltidx] = um[gidx - HALF_SCHEME * lda];
		}

		__syncthreads();
		
		if(ty < NPMLY) {
			uz = fd8_a1 * (smem[ltidy + 1][ltidx] - smem[ltidy - 1][ltidx])
			   + fd8_a2 * (smem[ltidy + 2][ltidx] - smem[ltidy - 2][ltidx])
			   + fd8_a3 * (smem[ltidy + 3][ltidx] - smem[ltidy - 2][ltidx])
			   + fd8_a4 * (smem[ltidy + 4][ltidx] - smem[ltidy - 4][ltidx]);		
			uzz = fd8_b0 *  smem[ltidy    ][ltidx]  
			    + fd8_b1 * (smem[ltidy + 1][ltidx] + smem[ltidy - 1][ltidx])
			    + fd8_b2 * (smem[ltidy + 2][ltidx] + smem[ltidy - 2][ltidx])
			    + fd8_b3 * (smem[ltidy + 3][ltidx] + smem[ltidy - 3][ltidx])
			    + fd8_b4 * (smem[ltidy + 4][ltidx] + smem[ltidy - 4][ltidx]); 
		}
		__syncthreads();
		
		smem[ltidy][ltidx] = uo[gidx];
		if(ty < HALF_SCHEME) {
			smem[ty][ltidx] = uo[gidx - HALF_SCHEME * lda];
		}

		__syncthreads();
		
		if(ty < NPMLY) {
			uz += fd8_a1 * (smem[ltidy + 1][ltidx] - smem[ltidy - 1][ltidx])
			   +  fd8_a2 * (smem[ltidy + 2][ltidx] - smem[ltidy - 2][ltidx])
			   +  fd8_a3 * (smem[ltidy + 3][ltidx] - smem[ltidy - 2][ltidx])
			   +  fd8_a4 * (smem[ltidy + 4][ltidx] - smem[ltidy - 4][ltidx]);		
			uzz += fd8_b0 *  smem[ltidy    ][ltidx]  
			    +  fd8_b1 * (smem[ltidy + 1][ltidx] + smem[ltidy - 1][ltidx])
			    +  fd8_b2 * (smem[ltidy + 2][ltidx] + smem[ltidy - 2][ltidx])
			    +  fd8_b3 * (smem[ltidy + 3][ltidx] + smem[ltidy - 3][ltidx])
			    +  fd8_b4 * (smem[ltidy + 4][ltidx] + smem[ltidy - 4][ltidx]); 
		}
		__syncthreads();
	
		uz /= 2.0f * dz;
		uzz /= 2.0f * dz * dz;
		
		// TO CONFIRM
		const int pmlidx = ty * lda + gtidx;

		if(ty < NPMLY) {
			pml2d_gpu_update(pmlidx, vel[gidx], ux, uxx, pmlzt);
		}		
	}

	if(blockIdx.y == gridDim.y - 1) {
		smem[ltidy][ltidx] = um[gidx];
		if(ty < HALF_SCHEME) {
			smem[ty + BLOCK_Y + HALF_SCHEME][ltidx] = um[gidx + BLOCK_Y * lda];
		}

		__syncthreads();
		
		if(ty >= blockDim.y - NPMLY) {
			uz = fd8_a1 * (smem[ltidy + 1][ltidx] - smem[ltidy - 1][ltidx])
			   + fd8_a2 * (smem[ltidy + 2][ltidx] - smem[ltidy - 2][ltidx])
			   + fd8_a3 * (smem[ltidy + 3][ltidx] - smem[ltidy - 2][ltidx])
			   + fd8_a4 * (smem[ltidy + 4][ltidx] - smem[ltidy - 4][ltidx]);		
			uzz = fd8_b0 *  smem[ltidy    ][ltidx]  
			    + fd8_b1 * (smem[ltidy + 1][ltidx] + smem[ltidy - 1][ltidx])
			    + fd8_b2 * (smem[ltidy + 2][ltidx] + smem[ltidy - 2][ltidx])
			    + fd8_b3 * (smem[ltidy + 3][ltidx] + smem[ltidy - 3][ltidx])
			    + fd8_b4 * (smem[ltidy + 4][ltidx] + smem[ltidy - 4][ltidx]); 
		}
		__syncthreads();
		
		smem[ltidy][ltidx] = uo[gidx];
		if(ty < HALF_SCHEME) {
			smem[ty + BLOCK_Y + HALF_SCHEME][ltidx] = uo[gidx + BLOCK_Y * lda];
		}

		__syncthreads();
		
		if(ty >= blockDim.y - NPMLY) {
			uz += fd8_a1 * (smem[ltidy + 1][ltidx] - smem[ltidy - 1][ltidx])
			   +  fd8_a2 * (smem[ltidy + 2][ltidx] - smem[ltidy - 2][ltidx])
			   +  fd8_a3 * (smem[ltidy + 3][ltidx] - smem[ltidy - 2][ltidx])
			   +  fd8_a4 * (smem[ltidy + 4][ltidx] - smem[ltidy - 4][ltidx]);		
			uzz += fd8_b0 *  smem[ltidy    ][ltidx]  
			    +  fd8_b1 * (smem[ltidy + 1][ltidx] + smem[ltidy - 1][ltidx])
			    +  fd8_b2 * (smem[ltidy + 2][ltidx] + smem[ltidy - 2][ltidx])
			    +  fd8_b3 * (smem[ltidy + 3][ltidx] + smem[ltidy - 3][ltidx])
			    +  fd8_b4 * (smem[ltidy + 4][ltidx] + smem[ltidy - 4][ltidx]); 
		}
		__syncthreads();
	
		uz /= 2.0f * dz;
		uzz /= 2.0f * dz * dz;
		
		// TO CONFIRM
		const int pmlidx = ty * lda + gtidx;

		if(ty >= blockDim.y - NPMLY) {
			pml2d_gpu_update(pmlidx, vel[gidx], ux, uxx, pmlzb);
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


