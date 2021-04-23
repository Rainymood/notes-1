---
title: "Number line jumps solution"
author: "Jan Meppe"
date: 2021-04-23
description: "Number line jumps solution"
type: technical_note
draft: false
---

## Problem statement

[Link to problem](https://www.hackerrank.com/challenges/kangaroo/problem)

## Lessons learned

* Set up equations and solve by hand
* Brute-force solutio and then think of edge cases

## Solution

```python
# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    if (x1-x2<0) and (v1-v2<0):
        return "NO"
    k = (x2-x1)/(v2-v1)
    if k.is_integer():
        return "YES"
    else: 
        return "NO"
    
```

Note: Fails on test case 10.