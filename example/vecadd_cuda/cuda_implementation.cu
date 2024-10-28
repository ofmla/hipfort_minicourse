#include <stdlib.h>
#include <stdio.h>
// cuda includes
#include "cuda_runtime.h"
#include "cuda.h"

__global__
void dxpy(int n, double *x, double *y, double *z)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = index; i < n; i += stride)
    z[i] = x[i] + y[i];
}

extern "C" void dxpy_cfcn(int N, double *x, double *y, double *out){
    
    double *d_x, *d_y, *d_out;   
    
    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_y, N*sizeof(double));
    cudaMalloc(&d_out, N*sizeof(double));
   
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);

   // Perform DAXPY on 1M elements
   dxpy<<<320, 256>>>(N, d_x, d_y, d_out);
   
   cudaMemcpy(out, d_out, N*sizeof(double), cudaMemcpyDeviceToHost);

   cudaFree(d_x);
   cudaFree(d_y);
   cudaFree(d_out);
}
