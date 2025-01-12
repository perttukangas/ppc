#include <vector>
#include <cmath>

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
  std::vector<double> means(ny, 0.0);
  for (int y = 0; y < ny; ++y)
  {
    for (int x = 0; x < nx; ++x)
    {
      means[y] += data[x + y * nx];
    }
    means[y] /= nx;
  }

  for (int i = 0; i < ny; ++i)
  {
    for (int j = 0; j <= i; ++j)
    {
      double numerator = 0.0;
      double denominator_i = 0.0;
      double denominator_j = 0.0;

      for (int x = 0; x < nx; ++x)
      {
        double diff_i = data[x + i * nx] - means[i];
        double diff_j = data[x + j * nx] - means[j];
        numerator += diff_i * diff_j;
        denominator_i += diff_i * diff_i;
        denominator_j += diff_j * diff_j;
      }

      double denominator = std::sqrt(denominator_i * denominator_j);
      if (denominator != 0.0)
      {
        result[i + j * ny] = static_cast<float>(numerator / denominator);
      }
      else
      {
        result[i + j * ny] = 0.0f;
      }
    }
  }
}
