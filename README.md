# implementing-parallel-SCAN-algorithm-in-CUDA
Assume an array of numbers. executing scan algorithm on this array will result an output array whose output are the following values:
-  The first value of the output array is the same as the first number of input array.
-  The second number of the output array is the summation of the first and the second value of the input array.
-  The third value of the output array is additin of first three values of the input array.
-  This continues to the end, when the last number of the output array is summation all values in input array.

Running the algorithm in this manner is called a serial algorithm. Using parallel processing techniques, we can design an algorithm which can calculate the output array in parallel. It is clearly faster the serial algorithm.
