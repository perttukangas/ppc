#include <algorithm>

using namespace std;

typedef unsigned long long data_t;

data_t median_of_three(data_t *data, int left, int right)
{
    int mid = (left + right) / 2;

    if (data[mid] < data[left])
    {
        swap(data[left], data[mid]);
    }
    if (data[right] < data[left])
    {
        swap(data[left], data[right]);
    }
    if (data[mid] < data[right])
    {
        swap(data[mid], data[right]);
    }

    return data[right];
}

void partition(data_t *data, int left, int right, int &i, int &j)
{
    data_t pivot = median_of_three(data, left, right);
    i = left;
    j = right;

    // Hoare
    while (i <= j)
    {
        while (data[i] < pivot)
        {
            ++i;
        }

        while (data[j] > pivot)
        {
            --j;
        }

        if (i <= j)
        {
            swap(data[i], data[j]);
            ++i;
            --j;
        }
    }
}

void quicksort(data_t *data, int left, int right)
{
    if (left < 0 || right < 0 || left >= right)
    {
        return;
    }

    if (right - left <= 1000000)
    {
        sort(data + left, data + right + 1);
        return;
    }

    int i;
    int j;
    partition(data, left, right, i, j);

#pragma omp task
    quicksort(data, left, j);

#pragma omp task
    quicksort(data, i, right);
}

void psort(int n, data_t *data)
{
#pragma omp parallel
    {
#pragma omp single
        {
            quicksort(data, 0, n - 1);
        }
    }
}