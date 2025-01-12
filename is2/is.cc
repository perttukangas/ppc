#include <vector>
#include <limits>

struct Result
{
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

static inline double rectSum(
    const std::vector<std::vector<std::vector<double>>> &sum,
    int c, int x0, int y0, int x1, int y1)
{
    // inclusive-exclusive
    return sum[c][y1][x1] - sum[c][y0][x1] - sum[c][y1][x0] + sum[c][y0][x0];
}

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data)
{
    // precompute sums and sum sq
    std::vector<std::vector<std::vector<double>>> sum(3, std::vector<std::vector<double>>(ny + 1, std::vector<double>(nx + 1, 0.0)));
    std::vector<std::vector<std::vector<double>>> sumSq(3, std::vector<std::vector<double>>(ny + 1, std::vector<double>(nx + 1, 0.0)));
    for (int c = 0; c < 3; ++c)
    {
        for (int y = 0; y < ny; ++y)
        {
            double rowSum = 0.0;
            double rowSumSq = 0.0;
            for (int x = 0; x < nx; ++x)
            {
                double val = data[c + 3 * x + 3 * nx * y];
                rowSum += val;
                rowSumSq += val * val;

                sum[c][y + 1][x + 1] = sum[c][y][x + 1] + rowSum;
                sumSq[c][y + 1][x + 1] = sumSq[c][y][x + 1] + rowSumSq;
            }
        }
    }

    // precompute total
    std::vector<double> total(3, 0.0), totalSq(3, 0.0);
    int n = ny * nx;
    for (int c = 0; c < 3; ++c)
    {
        total[c] = rectSum(sum, c, 0, 0, nx, ny);
        totalSq[c] = rectSum(sumSq, c, 0, 0, nx, ny);
    }

    double bestCost = std::numeric_limits<double>::max();
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

    // loop all rect
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
                        double sumIn = rectSum(sum, c, x0, y0, x1, y1);
                        double sumInSq = rectSum(sumSq, c, x0, y0, x1, y1);

                        double sumOut = total[c] - sumIn;
                        double sumOutSq = totalSq[c] - sumInSq;

                        double meanIn = sumIn / insideCount;
                        double meanOut = sumOut / outsideCount;

                        double errIn = sumInSq - 2.0 * meanIn * sumIn + meanIn * meanIn * insideCount;
                        double errOut = sumOutSq - 2.0 * meanOut * sumOut + meanOut * meanOut * outsideCount;

                        cost += errIn + errOut;
                        inCols[c] = meanIn;
                        outCols[c] = meanOut;
                    }

                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        result.y0 = y0;
                        result.x0 = x0;
                        result.y1 = y1;
                        result.x1 = x1;
                        for (int c = 0; c < 3; ++c)
                        {
                            result.inner[c] = inCols[c];
                            result.outer[c] = outCols[c];
                        }
                    }
                }
            }
        }
    }

    return result;
}
