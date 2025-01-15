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
  constexpr int elements_per_vector = 4;
  int vectors_per_row = (nx + elements_per_vector - 1) / elements_per_vector;

  vector<double4_t> diffs(ny * vectors_per_row);
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
    for (int vector = 0; vector < vectors_per_row; ++vector)
    {
      double4_t diff_block = {0.0, 0.0, 0.0, 0.0};
      for (int vector_i = 0; vector_i < elements_per_vector; ++vector_i)
      {
        int x = vector * elements_per_vector + vector_i;
        if (x < nx)
        {
          double diff = data[x + y * nx] - mean;
          diff_block[vector_i] = diff;
          sum += diff * diff;
        }
      }
      diffs[vector + y * vectors_per_row] = diff_block;
    }

    diffs_sqrt[y] = sqrt(sum);
  }

  for (int i = 0; i < ny; ++i)
  {
    for (int j = 0; j <= i; ++j)
    {
      double4_t vector = {0.0, 0.0, 0.0, 0.0};
      for (int current_vector = 0; current_vector < vectors_per_row; ++current_vector)
      {
        vector +=
            diffs[i * vectors_per_row + current_vector] *
            diffs[j * vectors_per_row + current_vector];
      }

      double top = (vector[0] + vector[1]) + (vector[2] + vector[3]);
      double bottom = diffs_sqrt[i] * diffs_sqrt[j];
      result[i + j * ny] = bottom != 0.0 ? static_cast<float>(top / bottom) : 0.0f;
    }
  }
}