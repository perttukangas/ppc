#include <vector>
#include <cmath>

using namespace std;

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
  int padded_size = block_amount * block_size;

  vector<double> diffs(ny * padded_size, 0.0);
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
    for (int x = 0; x < nx; ++x)
    {
      double diff = data[x + y * nx] - mean;
      diffs[x + y * padded_size] = diff;
      sum += diff * diff;
    }

    diffs_sqrt[y] = sqrt(sum);
  }

  double block[block_size] = {0.0};
  for (int i = 0; i < ny; ++i)
  {
    for (int j = 0; j <= i; ++j)
    {
      block[0] = block[1] = block[2] = block[3] = 0.0;
      for (int current_block = 0; current_block < block_amount; ++current_block)
      {
        for (int block_i = 0; block_i < block_size; ++block_i)
        {
          block[block_i] +=
              diffs[i * padded_size + current_block * block_size + block_i] *
              diffs[j * padded_size + current_block * block_size + block_i];
        }
      }

      double top = (block[0] + block[1]) + (block[2] + block[3]);
      double bottom = diffs_sqrt[i] * diffs_sqrt[j];
      result[i + j * ny] = bottom != 0.0 ? static_cast<float>(top / bottom) : 0.0f;
    }
  }
}
