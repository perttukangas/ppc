#include <vector>
#include <cmath>

using namespace std;

typedef float float8_t __attribute__((vector_size(8 * sizeof(float))));

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
  constexpr int elements_per_vector = 8;
  int vectors_per_row = (nx + elements_per_vector - 1) / elements_per_vector;

  constexpr int block_size = 6;
  int blocks_of_rows = (ny + block_size - 1) / block_size;
  int rows_after_padding = blocks_of_rows * block_size;

  vector<float8_t> diffs(rows_after_padding * vectors_per_row);
  vector<float> diffs_sqrt(ny, 0.0f);

// precompute all whats possible
#pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < ny; ++y)
  {
    float mean = 0.0f;
    for (int x = 0; x < nx; ++x)
    {
      mean += data[x + y * nx];
    }
    mean /= nx;

    float sum = 0.0f;
    for (int vector = 0; vector < vectors_per_row; ++vector)
    {
      float8_t diff_block = {};
      for (int vector_i = 0; vector_i < elements_per_vector; ++vector_i)
      {
        int x = vector * elements_per_vector + vector_i;
        if (x < nx)
        {
          float diff = data[x + y * nx] - mean;
          diff_block[vector_i] = diff;
          sum += diff * diff;
        }
      }
      diffs[vector + y * vectors_per_row] = diff_block;
    }

    diffs_sqrt[y] = sqrt(sum);
  }

#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < blocks_of_rows; ++i)
  {
    for (int j = 0; j <= i; ++j)
    {
      float8_t vector[block_size][block_size] = {};
      for (int current_vector = 0; current_vector < vectors_per_row; ++current_vector)
      {
        float8_t y0 = diffs[(j * block_size + 0) * vectors_per_row + current_vector];
        float8_t y1 = diffs[(j * block_size + 1) * vectors_per_row + current_vector];
        float8_t y2 = diffs[(j * block_size + 2) * vectors_per_row + current_vector];
        float8_t y3 = diffs[(j * block_size + 3) * vectors_per_row + current_vector];
        float8_t y4 = diffs[(j * block_size + 4) * vectors_per_row + current_vector];
        float8_t y5 = diffs[(j * block_size + 5) * vectors_per_row + current_vector];

        float8_t x0 = diffs[(i * block_size + 0) * vectors_per_row + current_vector];
        float8_t x1 = diffs[(i * block_size + 1) * vectors_per_row + current_vector];
        float8_t x2 = diffs[(i * block_size + 2) * vectors_per_row + current_vector];
        float8_t x3 = diffs[(i * block_size + 3) * vectors_per_row + current_vector];
        float8_t x4 = diffs[(i * block_size + 4) * vectors_per_row + current_vector];
        float8_t x5 = diffs[(i * block_size + 5) * vectors_per_row + current_vector];

        vector[0][0] += y0 * x0;
        vector[0][1] += y0 * x1;
        vector[0][2] += y0 * x2;
        vector[0][3] += y0 * x3;
        vector[0][4] += y0 * x4;
        vector[0][5] += y0 * x5;

        vector[1][0] += y1 * x0;
        vector[1][1] += y1 * x1;
        vector[1][2] += y1 * x2;
        vector[1][3] += y1 * x3;
        vector[1][4] += y1 * x4;
        vector[1][5] += y1 * x5;

        vector[2][0] += y2 * x0;
        vector[2][1] += y2 * x1;
        vector[2][2] += y2 * x2;
        vector[2][3] += y2 * x3;
        vector[2][4] += y2 * x4;
        vector[2][5] += y2 * x5;

        vector[3][0] += y3 * x0;
        vector[3][1] += y3 * x1;
        vector[3][2] += y3 * x2;
        vector[3][3] += y3 * x3;
        vector[3][4] += y3 * x4;
        vector[3][5] += y3 * x5;

        vector[4][0] += y4 * x0;
        vector[4][1] += y4 * x1;
        vector[4][2] += y4 * x2;
        vector[4][3] += y4 * x3;
        vector[4][4] += y4 * x4;
        vector[4][5] += y4 * x5;

        vector[5][0] += y5 * x0;
        vector[5][1] += y5 * x1;
        vector[5][2] += y5 * x2;
        vector[5][3] += y5 * x3;
        vector[5][4] += y5 * x4;
        vector[5][5] += y5 * x5;
      }

      for (int bi = 0; bi < block_size; ++bi)
      {
        int global_i = i * block_size + bi;
        if (global_i < ny)
        {
          float sqrt_i = diffs_sqrt[global_i];
          for (int bj = 0; bj < block_size; ++bj)
          {
            int global_j = j * block_size + bj;
            if (global_j < ny && global_j <= global_i)
            {
              float sum = (vector[bj][bi][0] + vector[bj][bi][1] +
                           vector[bj][bi][2] + vector[bj][bi][3] +
                           vector[bj][bi][4] + vector[bj][bi][5] +
                           vector[bj][bi][6] + vector[bj][bi][7]);
              float bottom = sqrt_i * diffs_sqrt[global_j];
              result[global_i + global_j * ny] = bottom != 0.0f ? sum / bottom : 0.0f;
            }
          }
        }
      }
    }
  }
}