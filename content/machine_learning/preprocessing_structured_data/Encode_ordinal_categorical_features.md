---
title: "Encode ordinal categorical"
author: "Jan Meppe"
date: 2021-04-09
description: "Encode ordinal categorical features"
type: technical_note
draft: false
---
* `OrdinalEncoder` is **very confusing** (so don't worry if you don't get it... it's confusing)
* Make sure you input a **list of lists**


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
```

## Create  data


```python
d = {'rating': ["first", "second", "third", "first", "second", "second"]}
df = pd.DataFrame(d)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>first</td>
    </tr>
    <tr>
      <th>1</th>
      <td>second</td>
    </tr>
    <tr>
      <th>2</th>
      <td>third</td>
    </tr>
    <tr>
      <th>3</th>
      <td>first</td>
    </tr>
    <tr>
      <th>4</th>
      <td>second</td>
    </tr>
    <tr>
      <th>5</th>
      <td>second</td>
    </tr>
  </tbody>
</table>
</div>



## Initialise transformer

Read carefully.

    list : categories[i] holds the categories expected in the ith column. The passed categories should not mix strings and numeric values, and should be sorted in case of numeric values.
    
`categories` is a list with each list having the expected categories in the ith column. In other words `categories = [mapping_col_1{}, mapping_col_2{}, ... ]`

My god this is confusing.


```python
# WRONG: categories = ["first", "second", "third"] # first = 0, second = 1, third = 2
categories = [["first", "second", "third"]] # NOTE: LIST OF LIST!!! 
ordinal_encoder = OrdinalEncoder(categories)
```

## Fit transformer


```python
X = df['rating'].to_numpy().reshape(-1,1)
ordinal_encoder.fit(X)
```




    OrdinalEncoder(categories=[['first', 'second', 'third']],
                   dtype=<class 'numpy.float64'>)



## Apply transformer


```python
ordinal_encoder.transform(X)
```




    array([[0.],
           [1.],
           [2.],
           [0.],
           [1.],
           [1.]])



## Get back labels from integers


```python
ordinal_encoder.inverse_transform(ordinal_encoder.transform(X))
```




    array([['first'],
           ['second'],
           ['third'],
           ['first'],
           ['second'],
           ['second']], dtype=object)




```python
categories = [["bad", "good", "neutral"]] 
enc = OrdinalEncoder(categories=categories)
X = np.array(["bad", "good", "neutral"]).reshape(-1,1)
enc.fit(X)
```




    OrdinalEncoder(categories=[['bad', 'good', 'neutral']],
                   dtype=<class 'numpy.float64'>)




```python
enc.fit_transform(X)
```




    array([[0.],
           [1.],
           [2.]])


