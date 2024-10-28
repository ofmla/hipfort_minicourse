#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void test_function_kernel(float *u, int nx, int ny, int nz,
                                     float hx, float hy, float hz) { 

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    // Exit if this thread is outside the boundary
    if (i >= nz ||
        j >= nx ||
        k >= ny)
        return;

    size_t pos = i + nz * (j +  nx * k);

    float c = 0.5;
    float x = j*hx;
    float y = k*hy;
    float z = i*hz;
    float Lx = nx*hx;
    float Ly = ny*hy;
    float Lz = nz*hz;
    u[pos] = c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz);
}

__global__ void laplacian_kernel(float *f, float *u, 
                                 float *cx, float *cy, float *cz, 
                                 int nx, int ny, int nz) {
    const int pad = 4;
    const int slice = nz * nx;
    int i = pad + (blockIdx.x * blockDim.x + threadIdx.x); 
    int j = pad + (blockIdx.y * blockDim.y + threadIdx.y); 
    int k = pad + (blockIdx.z * blockDim.z + threadIdx.z);
    float dsxxm = 0.0, dsyym = 0.0, dszzm = 0.0;
    
    // Compute the result of the stencil operation
    if(i < nz - pad){
        if(j < nx - pad){ 
            if(k < ny - pad){
            int dim_y = j * nz;
            dszzm += cz[0] * u[i+(dim_y)+(k*slice)];
            dszzm += cz[1] * (u[(i+1)+(dim_y)+(k*slice)] + u[(i-1)+(dim_y)+(k*slice)]);
            dszzm += cz[2] * (u[(i+2)+(dim_y)+(k*slice)] + u[(i-2)+(dim_y)+(k*slice)]);
            dszzm += cz[3] * (u[(i+3)+(dim_y)+(k*slice)] + u[(i-3)+(dim_y)+(k*slice)]);
            dszzm += cz[4] * (u[(i+4)+(dim_y)+(k*slice)] + u[(i-4)+(dim_y)+(k*slice)]);

            dsxxm += cz[0] * u[i+(dim_y)+(k*slice)]; 
            dsxxm += cx[1] * (u[i+((j+1)* nz)+(k*slice)] + u[i+((j-1)* nz)+(k*slice)]); 
            dsxxm += cx[2] * (u[i+((j+2)* nz)+(k*slice)] + u[i+((j-2)* nz)+(k*slice)]); 
            dsxxm += cx[3] * (u[i+((j+3)* nz)+(k*slice)] + u[i+((j-3)* nz)+(k*slice)]); 
            dsxxm += cx[4] * (u[i+((j+4)* nz)+(k*slice)] + u[i+((j-4)* nz)+(k*slice)]);

            dsyym += cy[0] * u[i+(dim_y)+(k*slice)];  
            dsyym += cy[1] * (u[i+(dim_y)+((k+1)*slice)] + u[i+(dim_y)+((k-1)*slice)]); 
            dsyym += cy[2] * (u[i+(dim_y)+((k+2)*slice)] + u[i+(dim_y)+((k-2)*slice)]); 
            dsyym += cy[3] * (u[i+(dim_y)+((k+3)*slice)] + u[i+(dim_y)+((k-3)*slice)]); 
            dsyym += cy[4] * (u[i+(dim_y)+((k+4)*slice)] + u[i+(dim_y)+((k-4)*slice)]);

            f[i+(dim_y)+(k*slice)] = dszzm+dsxxm+dsyym;
            } 
        }
    }
    
} 

extern "C"
{
  void test_function(dim3* grid, dim3* block, int shmem, hipStream_t stream, 
                     float *d_u, int nx, int ny, int nz, float hx, float hy, float hz)
  {
    //printf("launching kernel\n");
    hipLaunchKernelGGL((test_function_kernel), *grid, *block, shmem, stream, 
                       d_u, nx, ny, nz, hx, hy, hz);
  }

  void laplacian(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                 float *d_f, float *d_u, float *d_cx, float *d_cy, float *d_cz, int nx, int ny, int nz) 
  {
    hipLaunchKernelGGL((laplacian_kernel), *grid, *block, shmem, stream, 
                       d_f, d_u, d_cx, d_cy, d_cz, nx, ny, nz);
  } 
}
