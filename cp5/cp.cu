#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char *context)
{
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA error: " << context << ": "
              << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b)
{
  return (a + b - 1) / b;
}

__global__ void precompute(int ny, int nx, const float *data, float *diffs)
{
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= ny)
  {
    return;
  }

  float mean = 0.0f;
  for (int x = 0; x < nx; ++x)
  {
    mean += data[x + y * nx];
  }
  mean /= nx;

  float sum = 0.0f;
  for (int x = 0; x < nx; ++x)
  {
    float diff = data[x + y * nx] - mean;
    diffs[x + y * nx] = diff;
    sum += diff * diff;
  }

  for (int x = 0; x < nx; ++x)
  {
    diffs[x + y * nx] /= sqrt(sum);
  }
}

__global__ void compute(int ny, int nx, float *diffs, float *result)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= ny || j > i)
  {
    return;
  }

  double sum = 0.0;
  for (int x = 0; x < nx; ++x)
  {
    sum += diffs[x + i * nx] * diffs[x + j * nx];
  }

  result[i + j * ny] = sum;
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
  float *d_data, *d_result, *d_diffs;

  size_t data_size = ny * nx * sizeof(float);
  size_t result_size = ny * ny * sizeof(float);
  size_t diffs_size = ny * nx * sizeof(float);

  CHECK(cudaMalloc((void **)&d_data, data_size));
  CHECK(cudaMalloc((void **)&d_result, result_size));
  CHECK(cudaMalloc((void **)&d_diffs, diffs_size));

  CHECK(cudaMemset(d_data, 0, data_size));
  CHECK(cudaMemset(d_result, 0, result_size));
  CHECK(cudaMemset(d_diffs, 0, diffs_size));

  CHECK(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));

  {
    dim3 dim_block(32);
    dim3 dim_grid(divup(ny, dim_block.x));
    precompute<<<dim_grid, dim_block>>>(ny, nx, d_data, d_diffs);
    CHECK(cudaGetLastError());
  }

  {
    dim3 dim_block(16, 16, 1);
    dim3 dim_grid(divup(ny, dim_block.x), divup(ny, dim_block.y), 1);
    compute<<<dim_grid, dim_block>>>(ny, nx, d_diffs, d_result);
    CHECK(cudaGetLastError());
  }

  CHECK(cudaMemcpy(result, d_result, result_size, cudaMemcpyDeviceToHost));

  CHECK(cudaFree(d_data));
  CHECK(cudaFree(d_result));
  CHECK(cudaFree(d_diffs));
}