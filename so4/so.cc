#include <algorithm>

using namespace std;

typedef unsigned long long data_t;

void copy_array(data_t *data, int left, int right, data_t *temp)
{
#pragma omp parallel for
    for (int k = left; k < right; k++)
    {
        temp[k] = data[k];
    }
}

void top_down_merge(data_t *data, int left, int mid, int right, data_t *temp)
{
    int i = left;
    int j = mid;

    for (int k = left; k < right; k++)
    {
        if (i < mid && (j >= right || data[i] <= data[j]))
        {
            temp[k] = data[i];
            i++;
        }
        else
        {
            temp[k] = data[j];
            j++;
        }
    }
}

void top_down_split_merge(data_t *temp, int left, int right, data_t *data)
{
    if (right - left <= 1000000)
    {
        sort(data + left, data + right);
        return;
    }

    int mid = (right + left) / 2;

#pragma omp taskgroup
    {
#pragma omp task
        top_down_split_merge(data, left, mid, temp);
#pragma omp task
        top_down_split_merge(data, mid, right, temp);
    }

    top_down_merge(temp, left, mid, right, data);
}

void psort(int n, data_t *data)
{
    data_t *temp = new data_t[n];
    copy_array(data, 0, n, temp);

#pragma omp parallel
    {
#pragma omp single
        {
            // Same impl as in wikipedia, but with temp and data swapped here so no need for copy
            top_down_split_merge(temp, 0, n, data);
        }
    }
    delete[] temp;
}