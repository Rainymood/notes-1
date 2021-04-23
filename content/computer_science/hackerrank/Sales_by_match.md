---
title: "Sales by Match solution"
author: "Jan Meppe"
date: 2021-04-14
description: "Sales by Match solution"
type: technical_note
draft: false
---
## Problem

[link](https://www.hackerrank.com/challenges/sock-merchant/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=warmup)

There is a large pile of socks that must be paired by color. Given an array of integers representing the color of each sock, determine how many pairs of socks with matching colors there are.

Example


There is one pair of color  and one of color . There are three odd socks left, one of each color. The number of pairs is .

Function Description

Complete the sockMerchant function in the editor below.

sockMerchant has the following parameter(s):

int n: the number of socks in the pile
int ar[n]: the colors of each sock
Returns

int: the number of pairs
Input Format

The first line contains an integer , the number of socks represented in .
The second line contains  space-separated integers, , the colors of the socks in the pile.

Constraints

 where 
Sample Input

STDIN                       Function
-----                       --------
9                           n = 9
10 20 20 10 10 30 50 10 20  ar = [10, 20, 20, 10, 10, 30, 50, 10, 20]
Sample Output

3
Explanation

sock.png

There are three pairs of socks.


STDIN                       Function
-----                       --------
9                           n = 9
10 20 20 10 10 30 50 10 20  ar = [10, 20, 20, 10, 10, 30, 50, 10, 20]

## Imports


```python
from collections import Counter
```

## Solution

The solution here is to know about the existence of the `Counter` object.


```python
n = 9
ar = [10, 20, 20, 10, 10, 30, 50, 10, 20]

def sockMerchant(n, ar):
    return sum([Counter(ar)[x] // 2 for x in Counter(ar)])

print(sockMerchant(n, ar))
```

    3


## Counter object makes this easy


```python
Counter(ar)
```




    Counter({10: 4, 20: 3, 30: 1, 50: 1})


