//Do NOT MODIFY THIS FILE
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gpuerrors.h"
#include "scan2.h"

//-----------------------------------------------------------------------------
void fill(float* data, int size) {			// to fill array with values in range [-2,+2]
    for (int i=0; i<size; ++i)
        data[i] = (float) 1.0*((rand() % 5)-2);
}

double calc_mse (float* data1, float* data2, int size) {	// calculate error by (  )^2
	double mse = 0.0;
	int i; 
	double e = 0.0;
	for (i=0; i<size; i++) {
		e = data1[i] - data2[i];
		e = e * e;
		mse += e;
		//if (e != 0)
		//	printf("%d\n", i);
	}
	mse = mse / ((double)size);
	return mse;
}
//-----------------------------------------------------------------------------
void cpuKernel (float* a, float* c, int n) {		// calculate scan algorithm  
	int i = 0;
	float sum = 0;
	for (i = 0; i < n; i += 1){
		//c[i] = sum;			// to have exclusive scan
		sum += (a[i]);
		c[i] = sum;			// to have inclusive scan
	}
	return;
}
//-----------------------------------------------------------------------------
int main ( int argc, char** argv) {
	   
	float* a;
	float* c_serial;
	float* c;	
	
	int m = 5; 
	int n = 32;

	if (argc > 1){
		m = atoi(argv[1]);
		n = (1 << m);
	}

	a        = (float*)malloc(n * sizeof(float));
	c_serial = (float*)malloc(n * sizeof(float));
	c        = (float*)malloc(n * sizeof(float));
				
	srand(0);
	fill(a, n);

	cpuKernel (a, c_serial, n);
		
	clock_t t0 = clock(); 
	gpuKernel (a, c, n);
	clock_t t1 = clock();
		
	float mse;
	mse = calc_mse( c_serial, c, n );
	
	printf("n=%d\t time=%06ld ms mse=%f\n",n, (t1-t0)/1000, mse);	
	//printf("%d\t%d\t%d\t%d\t%d\t", c[0],c[1],c[2],c[3],c[4]);

	free(a);
	free(c_serial);
	free(c);
	return 0;
}
//Do NOT MODIFY THIS FILE

