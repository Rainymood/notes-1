---
title: "Making anagrams solution"
author: "Jan Meppe"
date: 2021-04-23
description: "Making anagrams solution"
type: technical_note
draft: false
---

## Problem Statement

[Link to problem statement](https://www.hackerrank.com/challenges/ctci-making-anagrams/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=strings)

## Lessons learned

* Use python built-in libraries like `collections.Counter` (very useful tool)
* Brute force a problem on paper first with a small example

## Solution

```python
def makeAnagram(a, b):
    from collections import Counter
    counter_a = Counter(a)
    counter_b = Counter(b)
    counter_a.subtract(counter_b)
    return sum(abs(i) for i in counter_a.values())
```