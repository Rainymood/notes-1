---
title: "Arrays: Left Rotation solution"
author: "Jan Meppe"
date: 2021-04-12
description: "Arrays: Left Rotation solution"
type: technical_note
draft: false
---

## Problem

[pdf](/pdf/ctci-array-left-rotation-English.pdf)

See [this link](https://community.rstudio.com/t/is-there-a-place-we-can-put-non-blog-files-pdf-files-in-blogdown/10138/4) on how to link to static files in Hugo.

## Solution

```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the rotLeft function below.
def rotLeft(a, d):
    for i in range(d):
        a.append(a.pop(0))
    return a

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nd = input().split()

    n = int(nd[0])

    d = int(nd[1])

    a = list(map(int, input().rstrip().split()))

    result = rotLeft(a, d)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```