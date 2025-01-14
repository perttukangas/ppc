#include <vector>
#include <cmath>

using namespace std;

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

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
  constexpr int block_size = 4;
  int block_amount = (nx + block_size - 1) / block_size;

  vector<double4_t> diffs(ny * block_amount);
  vector<double> diffs_sqrt(ny, 0.0);

  // precompute all whats possible
  for (int y = 0; y < ny; ++y)
  {
    double mean = 0.0;
    for (int x = 0; x < nx; ++x)
    {
      mean += data[x + y * nx];
    }
    mean /= nx;

    double sum = 0.0;
    for (int block = 0; block < block_amount; ++block)
    {
      double4_t diff_block = {0.0, 0.0, 0.0, 0.0};
      for (int i = 0; i < block_size; ++i)
      {
        int x = block * block_size + i;
        if (x < nx)
        {
          double diff = data[x + y * nx] - mean;
          diff_block[i] = diff;
          sum += diff * diff;
        }
      }
      diffs[block + y * block_amount] = diff_block;
    }

    diffs_sqrt[y] = sqrt(sum);
  }

  for (int i = 0; i < ny; ++i)
  {
    for (int j = 0; j <= i; ++j)
    {
      double4_t block = {0.0, 0.0, 0.0, 0.0};
      for (int current_block = 0; current_block < block_amount; ++current_block)
      {
        block +=
            diffs[i * block_amount + current_block] *
            diffs[j * block_amount + current_block];
      }

      double top = (block[0] + block[1]) + (block[2] + block[3]);
      double bottom = diffs_sqrt[i] * diffs_sqrt[j];
      result[i + j * ny] = bottom != 0.0 ? static_cast<float>(top / bottom) : 0.0f;
    }
  }
}