struct Result
{
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1)
{
    double temp[3] = {0.0, 0.0, 0.0};

    for (int x = x0; x < x1; x++)
    {
        for (int y = y0; y < y1; y++)
        {
            for (int c = 0; c < 3; c++)
            {
                temp[c] += data[c + 3 * x + 3 * nx * y];
            }
        }
    }

    int rectSize = (x1 - x0) * (y1 - y0);
    for (int c = 0; c < 3; c++)
    {
        temp[c] /= rectSize;
    }

    Result result;
    for (int c = 0; c < 3; c++)
    {
        result.avg[c] = static_cast<float>(temp[c]);
    }

    return result;
}
