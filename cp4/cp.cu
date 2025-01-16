#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

__global__ void precompute(int ny, int nx, const float *data, float *diffs, float *diffs_sqrt) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= ny) return;

  float mean = 0.0f;
  for (int x = 0; x < nx; ++x) {
    mean += data[x + y * nx];
  }
  mean /= nx;

  float sum = 0.0f;
  for (int x = 0; x < nx; ++x) {
    float diff = data[x + y * nx] - mean;
    diffs[x + y * nx] = diff;
    sum += diff * diff;
  }

  diffs_sqrt[y] = sqrt(sum);
}

__global__ void compute(int ny, int nx, float *diffs, float *diffs_sqrt, float *result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (i >= ny || j > i) return;
  
  double top = 0.0;
  for (int x = 0; x < nx; ++x) {
    top += diffs[x + i * nx] * diffs[x + j * nx];
  }
  
  double bottom = diffs_sqrt[i] * diffs_sqrt[j];
  result[i + j * ny] = bottom != 0.0 ? top / bottom : 0.0f;
}

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result)
{
  float *d_data, *d_result;
  float *d_diffs, *d_diffs_sqrt;

  size_t data_size = ny * nx * sizeof(float);
  size_t result_size = ny * ny * sizeof(float);
  size_t diffs_sqrt_size = ny * sizeof(float);

  CHECK(cudaMalloc((void**)&d_data, data_size));
  CHECK(cudaMalloc((void**)&d_result, result_size));
  CHECK(cudaMalloc((void**)&d_diffs, data_size));
  CHECK(cudaMalloc((void**)&d_diffs_sqrt, diffs_sqrt_size));

  CHECK(cudaMemset(d_data, 0, data_size));
  CHECK(cudaMemset(d_result, 0, result_size));
  CHECK(cudaMemset(d_diffs, 0, data_size));
  CHECK(cudaMemset(d_diffs_sqrt, 0, diffs_sqrt_size));

  CHECK(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));

  // warp size = 32 threads (threads should be multiple of one warp)
  // max threads per block = 1024

  // ny = 85
  // blocks per dim = divup(85,32) = 3
  // total blocks = 3
  // threads per block = 32
  // total threads = 3 * 32 = 96
  dim3 dim_grid(divup(ny, 32));
  dim3 block_dim(32);
  precompute<<<dim_grid, block_dim>>>(ny, nx, d_data, d_diffs, d_diffs_sqrt);
  CHECK(cudaGetLastError());

  // ny = 34
  // blocks per dim = divup(34,16) = 3 
  // total blocks = 3 * 3 = 9
  // threads per block = 16 * 16 = 256
  // total threads = 9 * 256 = 2304
  dim3 dim_grid2(divup(ny, 16), divup(ny, 16), 1);
  dim3 block_dim2(16, 16, 1);
  compute<<<dim_grid2, block_dim2>>>(ny, nx, d_diffs, d_diffs_sqrt, d_result);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(result, d_result, result_size, cudaMemcpyDeviceToHost));

  CHECK(cudaFree(d_data));
  CHECK(cudaFree(d_result));
  CHECK(cudaFree(d_diffs));
  CHECK(cudaFree(d_diffs_sqrt));
}