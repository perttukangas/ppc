#include <algorithm>
#include <vector>

using namespace std;

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
void mf(int ny, int nx, int hy, int hx, const float *in, float *out)
{
  vector<float> window((2 * hx + 1) * (2 * hy + 1));

  for (int y = 0; y < ny; ++y)
  {
    for (int x = 0; x < nx; ++x)
    {

      int y_start = max(0, y - hy);
      int y_end = min(ny, y + hy + 1);
      int x_start = max(0, x - hx);
      int x_end = min(nx, x + hx + 1);

      int window_size = 0;
      for (int wy = y_start; wy < y_end; ++wy)
      {
        for (int wx = x_start; wx < x_end; ++wx)
        {
          window[window_size++] = in[wx + wy * nx];
        }
      }

      int mid = window_size / 2;
      nth_element(window.begin(), window.begin() + mid, window.begin() + window_size);

      if (window_size % 2 == 0)
      {
        // If n = 2k, then the median of a is (x _k + x _k+1)/2
        float right = window[mid];
        nth_element(window.begin(), window.begin() + mid - 1, window.begin() + window_size);
        float left = window[mid - 1];
        out[x + y * nx] = (left + right) / 2;
      }
      else
      {
        // If n = 2k + 1, then the median of a is x _k+1
        out[x + y * nx] = window[mid];
      }
    }
  }
}