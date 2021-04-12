---
title: "2D array hourglass"
author: "Jan Meppe"
date: 2021-04-12
description: "2D array hourglass"
type: technical_note
draft: false
---

## Problem

![link](2d-array-English.pdf)

## Solution

```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the hourglassSum function below.
def hourglassSum(arr):
    count = -63
    for i in range(4):
        for j in range(4):
            hourglass_sum = arr[i][j] + arr[i][j+1] + arr[i][j+2] + arr[i+1][j+1] + arr[i+2][j] + arr[i+2][j+1] + arr[i+2][j+2]
            
            if hourglass_sum > count:
                count = hourglass_sum
    return count

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    arr = []

    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))

    result = hourglassSum(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
```

## Main idea

* Draw the problem
* Brute-force loop through the array, calculate, and keep track of the maximum value

## Lessons

* The first step you want to do is brute-force a solution
* Use a counter to keep track of the target value
