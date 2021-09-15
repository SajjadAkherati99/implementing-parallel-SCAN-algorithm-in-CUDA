// ONLY MODIFY THIS FILE

#include "scan2.h"
#include "gpuerrors.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// you may define other parameters here!

int THREADS_EACH_BLOCK = 512;
int ELEMENTS_EACH_BLOCK = 1024;
int SHARED_SIZE = 1024 * sizeof(float);

// you may define other macros here!

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

// you may define other functions here!

__global__ void block_scan(float *cc, float *aa, int n)
{
	extern __shared__ float temp[];

	int ai = tx;		   // By using this thread indexing, we can avod bank conflicts when we want ...
	int bi = tx + (n / 2); // to read from aa memory to extern temp memory.
	
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai); // this index are used to avoid bank conflicts ...
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi); // we has stored them to retun the result in last step.

	temp[ai + bankOffsetA] = aa[ai]; // this approch help us to avoid bank conflict, when we want to travel ...
	temp[bi + bankOffsetB] = aa[bi]; // the tree of the array.
	

	int offset = 1; // an offset value to travell each step of log(n)
	for (int d = (n/2); d > 0; d /= 2) 
	{
		__syncthreads();
		if (tx < d)
		{
			int ai = offset * (2 * tx + 1) - 1;
			int bi = offset * (2 * tx + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (tx == 0) {
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; // clear the last element (inclusive algorithm)
	}

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan (the second logn steps)
	{
		offset /= 2;
		__syncthreads();
		if (tx < d)
		{
			int ai = offset * (2 * tx + 1) - 1;
			int bi = offset * (2 * tx + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai); // to avoid conflicts
			bi += CONFLICT_FREE_OFFSET(bi);

			int hold = temp[ai];
			temp[ai] = temp[bi]; // left child to right child and righ vhild addition with left child ...
			temp[bi] += hold; 	 // in right child. 
		}
	}
	__syncthreads();

	cc[ai] = temp[ai + bankOffsetA]; // save the results
	cc[bi] = temp[bi + bankOffsetB];
}

__global__ void grid_scan(float *cc, float *aa, int n, float *sum_arr) {
	extern __shared__ float temp[];

	int blockOffset = bx * n;
	
	int ai = tx;
	int bi = tx + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = aa[blockOffset + ai];
	temp[bi + bankOffsetB] = aa[blockOffset + bi];

	int offset = 1;
	for (int d = (n/2); d > 0; d /= 2) // first logn steps
	{
		__syncthreads();
		if (tx < d)
		{
			int ai = offset * (2 * tx + 1) - 1;
			int bi = offset * (2 * tx + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (tx == 0) { 
		sum_arr[bx] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)]; // save the last element of
																 // eahc block inclusive scan in sum array.
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;			 // take tha last element of each block scan
																 // to complete the second step of inclusive ...
																 // scan algorithm.
	} 
	
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset /= 2;
		__syncthreads();
		if (tx < d)
		{
			int ai = offset * (2 * tx + 1) - 1;
			int bi = offset * (2 * tx + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int hold = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += hold;
		}
	}
	__syncthreads();

	cc[blockOffset + ai] = temp[ai + bankOffsetA];
	cc[blockOffset + bi] = temp[bi + bankOffsetB];
}

// this module is used to add the last elemnen of each block scan to each element of the next one.
__global__ void add_last_element_scan(float *cc_inc, float *sum_arr, int n) {
	int blockOffset = n * bx;

	cc_inc[blockOffset + tx] += sum_arr[bx];
}

// this module is used to complete the scan algorithm for exclusive one by adding each element of the ...
// main array the main one.
__global__ void add_exc(float *scanned_inc, float *aa, int n, float dc) {
	int i = bx*n + tx;
	
	scanned_inc[i] += (aa[i]+dc);
}


// this module is used for large length of the input for scan. when input is bigger that 1024, this module
// is usef to scan in multiple block.
void my_scan(float *cc, float *aa, int n) {
	const int blocks_num = n / ELEMENTS_EACH_BLOCK;

	float *block_sum, *block_sum_scan;
	HANDLE_ERROR(cudaMalloc((void **)&block_sum, blocks_num * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void **)&block_sum_scan, blocks_num * sizeof(float)));


	grid_scan <<<blocks_num, THREADS_EACH_BLOCK, 2*SHARED_SIZE >>>(cc, aa, ELEMENTS_EACH_BLOCK, block_sum);

	const int block_sum_scan_size = blocks_num / 2;
	if (block_sum_scan_size > THREADS_EACH_BLOCK) {
		my_scan(block_sum_scan, block_sum, blocks_num);
	}
	else {
		block_scan <<< 1, blocks_num/2, 2*blocks_num*sizeof(float) >>> (block_sum_scan, block_sum, blocks_num);
	}

	add_last_element_scan <<<blocks_num, ELEMENTS_EACH_BLOCK>>> (cc, block_sum_scan, ELEMENTS_EACH_BLOCK);

	cudaFree(block_sum);
	cudaFree(block_sum_scan);
}


// when ths size of the input is bigger than or equal to 2^26,
// we need to copy 2^25 part of a array and c array in ram.
// this is because of our GPU chain.
void gpuKernel_Large(float* a, float* c,int n, int offset, float dc) {

	int m = n>>26;
	if(m == 1){
		float* aa;
		float* cc;
		float* a_pointer = a + offset;
		float* c_pointer = c + offset;
		
		HANDLE_ERROR(cudaMalloc((void**)&aa, (n/2) * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cc, (n/2) * sizeof(float)));
		
		
		HANDLE_ERROR(cudaMemcpy(aa, a_pointer, (n/2) * sizeof(float), cudaMemcpyHostToDevice));
		my_scan(cc, aa, n/2);
		const int blocks = n/(2*ELEMENTS_EACH_BLOCK);
		add_exc <<< blocks, ELEMENTS_EACH_BLOCK >>> (cc, aa, ELEMENTS_EACH_BLOCK, dc);
		HANDLE_ERROR(cudaMemcpy(c_pointer, cc, (n/2) * sizeof(float), cudaMemcpyDeviceToHost));
		
		dc = c[n/2-1 + offset];
		a_pointer = a_pointer + (n/2);
		c_pointer = c_pointer + (n/2);
		
		HANDLE_ERROR(cudaMemcpy(aa, a_pointer, (n/2) * sizeof(float), cudaMemcpyHostToDevice));
		my_scan(cc, aa, n/2);
		add_exc <<< blocks, ELEMENTS_EACH_BLOCK >>> (cc, aa, ELEMENTS_EACH_BLOCK, dc);
		HANDLE_ERROR(cudaMemcpy(c_pointer, cc, (n/2) * sizeof(float), cudaMemcpyDeviceToHost));
		
		HANDLE_ERROR(cudaFree(aa));
		HANDLE_ERROR(cudaFree(cc));
	}
	else{
		gpuKernel_Large(a, c, n/2, offset, dc);
		dc = c[offset + n/2-1];
		gpuKernel_Large(a, c, n/2, offset+n/2, dc);
	}

}	


void gpuKernel(float* a, float* c,int n) { 

	if(n>>26 == 0){
		float* aa;
		float* cc;
		
		HANDLE_ERROR(cudaMalloc((void**)&aa, n * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cc, n * sizeof(float)));
		
		HANDLE_ERROR(cudaMemcpy(aa, a, n * sizeof(float), cudaMemcpyHostToDevice));
		
		if (n > ELEMENTS_EACH_BLOCK) {
			my_scan(cc, aa, n);
		}
		else {
			block_scan <<< 1, (n+1)/2, 2*n*sizeof(float) >>> (cc, aa, n);
		}
		
		const int blocks = n/ELEMENTS_EACH_BLOCK;
		add_exc <<< blocks, ELEMENTS_EACH_BLOCK >>> (cc, aa, ELEMENTS_EACH_BLOCK, 0);

		HANDLE_ERROR(cudaMemcpy(c, cc, n * sizeof(float), cudaMemcpyDeviceToHost));

		HANDLE_ERROR(cudaFree(aa));
		HANDLE_ERROR(cudaFree(cc));
	}
	else{
		gpuKernel_Large(a, c, n, 0, 0);
	}
}

