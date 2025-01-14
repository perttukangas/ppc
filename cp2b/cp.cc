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
  vector<double> diffs(ny * nx, 0.0);
  vector<double> diffs_sqrt(ny, 0.0);

#pragma omp parallel for schedule(dynamic, 1)
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
      diffs[x + y * nx] = diff;
      sum += diff * diff;
    }

    diffs_sqrt[y] = sqrt(sum);
  }

#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < ny; ++i)
  {
    for (int j = 0; j <= i; ++j)
    {
      double top = 0.0;

      for (int x = 0; x < nx; ++x)
      {
        top += diffs[x + i * nx] * diffs[x + j * nx];
      }

      double bottom = diffs_sqrt[i] * diffs_sqrt[j];
      result[i + j * ny] = bottom != 0.0 ? static_cast<float>(top / bottom) : 0.0f;
    }
  }
}
