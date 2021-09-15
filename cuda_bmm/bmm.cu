//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!
#include "bmm.h"
#include <math.h>

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// TILEX and TILEY are used to set the number of threads in a CUDA block 
#define TILEX 32
#define TILEY 16

// you may define other parameters here!
#define TILEYX (((TILEX) < (8)) ? 16 : (TILEY == 4)? 64: 128)

// you may define other macros here!
// you may define other functions here!

dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(n/TILEX,n/TILEY);
	return dimGrid;
}
dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX,TILEY);
	return dimBlock;
}
__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {

	// write your GPU kernel function here
	__shared__ float ads[TILEY][TILEYX];
	__shared__ float bds[TILEYX][TILEX];
	
	int Row = by * TILEY + ty;
	int Col = bx * TILEX + tx;
	
	float s = 0;
	
	for (int i = 0; i < n/TILEYX; ++i) {
		if (TILEYX >= max(TILEX, TILEY)){
			if (TILEX >= TILEY){
				for (int rn = 0; rn < TILEYX/TILEY; rn++){
					bds[ty + rn*TILEY][tx] = bd[(i*TILEYX + ty+rn*TILEY)*n + Col];
				} 
				for (int cn = 0; cn < TILEYX/TILEX; cn++){
					ads[ty][tx + cn*TILEX] = ad[Row*n + i*TILEYX + tx + cn*TILEX]; 
				}
			}
			else{
				for (int cn = 0; cn < TILEYX/TILEX; cn++){
					ads[ty][tx + cn*TILEX] = ad[Row*n + i*TILEYX + tx + cn*TILEX]; 
				}
				for (int rn = 0; rn < TILEYX/TILEY; rn++){
					bds[ty + rn*TILEY][tx] = bd[(i*TILEYX + ty+rn*TILEY)*n + Col];
				} 
			}
		}
		else if (TILEYX <= min(TILEX, TILEY)){
			if (ty < TILEYX )
				bds[ty][tx] = bd[(i*TILEYX + ty)*n + Col];
			if(tx < TILEYX)
				ads[ty][tx] = ad[Row*n + i*TILEYX + tx]; 
		}
		
		else {
			if (TILEX < TILEY){
				for (int cn = 0; cn < TILEYX/TILEX; cn++){
					ads[ty][tx + cn*TILEX] = ad[Row*n + i*TILEYX + tx + cn*TILEX]; 
				}
				if (ty < TILEYX )
						bds[ty][tx] = bd[(i*TILEYX + ty)*n + Col];
			}
			else{
				for (int rn = 0; rn < TILEYX/TILEY; rn++){
					bds[ty + rn*TILEY][tx] = bd[(i*TILEYX + ty+rn*TILEY)*n + Col];
				} 
				if(tx < TILEYX)
					ads[ty][tx] = ad[Row*n + i*TILEYX + tx]; 
			}
		}
		
		__syncthreads();
		for (int j = 0; j < TILEYX; ++j) {
			s += ads[ty][j] * bds[j][tx];
		}
		__syncthreads();
	}
	
	cd[Row*n + Col] = s;
	
}
