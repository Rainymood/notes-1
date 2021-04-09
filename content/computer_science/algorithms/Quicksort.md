---
title: "Quicksort"
author: "Jan Meppe"
date: 2021-04-09
description: "Quicksort"
type: technical_note
draft: false
---
# Quicksort

Implementation of the quicksort algorithm. This divide and conquer algorithm works by splitting up the problem by: 

* Splitting up the problem in a base case (empty or element 1)
* Subdividing the problem until you end up in one of the base cases

Solution is not optimal, construction of lesser and greater could be done in place I believe.


```python
def quicksort(arr):
    # base case [] or [.]
    if len(arr) < 2:
        return arr
    # otherwise choose pivot and recurse
    pivot = arr[0]
    lesser = [i for i in arr[1:] if i <= pivot]
    greater = [i for i in arr[1:] if i >= pivot]

    return quicksort(lesser) + [pivot] + quicksort(greater)


quicksort([1, 2, 1, 0, 1, 2, 3, 4])
```




    [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4]


