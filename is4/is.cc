#include <vector>
#include <limits>

using namespace std;

struct Result
{
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

int idx(int c, int y, int x, int nx, int ny)
{
    return c * (ny + 1) * (nx + 1) + y * (nx + 1) + x;
}

static inline double rectSum(
    const vector<double> &sum,
    int c, int x0, int y0, int x1, int y1, int nx, int ny)
{
    return sum[idx(c, y1, x1, nx, ny)] - sum[idx(c, y0, x1, nx, ny)] -
           sum[idx(c, y1, x0, nx, ny)] + sum[idx(c, y0, x0, nx, ny)];
}

Result segment(int ny, int nx, const float *data)
{
    vector<double> sum((ny + 1) * (nx + 1) * 3, 0.0);

#pragma omp for schedule(static, 1)
    for (int c = 0; c < 3; ++c)
    {
        for (int y = 0; y < ny; ++y)
        {
            double rowSum = 0.0;
            for (int x = 0; x < nx; ++x)
            {
                double val = data[c + 3 * x + 3 * nx * y];
                rowSum += val;

                sum[idx(c, y + 1, x + 1, nx, ny)] = sum[idx(c, y, x + 1, nx, ny)] + rowSum;
            }
        }
    }

    // precompute total
    vector<double> total(3, 0.0);
    int n = ny * nx;

#pragma omp for schedule(static, 1)
    for (int c = 0; c < 3; ++c)
    {
        total[c] = rectSum(sum, c, 0, 0, nx, ny, nx, ny);
    }

    double bestCost = numeric_limits<double>::max();
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

#pragma omp parallel
    {
        double localBestCost = numeric_limits<double>::max();
        Result localResult{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

#pragma omp for schedule(static, 1)
        for (int y0 = 0; y0 < ny; ++y0)
        {
            for (int y1 = y0 + 1; y1 <= ny; ++y1)
            {
                for (int x0 = 0; x0 < nx; ++x0)
                {
                    for (int x1 = x0 + 1; x1 <= nx; ++x1)
                    {
                        int insideCount = (y1 - y0) * (x1 - x0);
                        int outsideCount = n - insideCount;
                        if (outsideCount <= 0)
                            continue;

                        double cost = 0.0;
                        float inCols[3], outCols[3];

                        for (int c = 0; c < 3; ++c)
                        {
                            double sumIn = rectSum(sum, c, x0, y0, x1, y1, nx, ny);
                            double sumInSq = sumIn * sumIn;

                            double sumOut = total[c] - sumIn;
                            double sumOutSq = (total[c] * total[c]) - sumInSq;

                            double meanIn = sumIn / insideCount;
                            double meanOut = sumOut / outsideCount;

                            double errIn = sumInSq - 2.0 * meanIn * sumIn + meanIn * meanIn * insideCount;
                            double errOut = sumOutSq - 2.0 * meanOut * sumOut + meanOut * meanOut * outsideCount;

                            cost += errIn + errOut;
                            inCols[c] = meanIn;
                            outCols[c] = meanOut;
                        }

                        if (cost < localBestCost)
                        {
                            localBestCost = cost;
                            localResult.y0 = y0;
                            localResult.x0 = x0;
                            localResult.y1 = y1;
                            localResult.x1 = x1;
                            for (int c = 0; c < 3; ++c)
                            {
                                localResult.inner[c] = inCols[c];
                                localResult.outer[c] = outCols[c];
                            }
                        }
                    }
                }
            }
        }

#pragma omp critical
        {
            if (localBestCost < bestCost)
            {
                bestCost = localBestCost;
                result = localResult;
            }
        }
    }

    return result;
}