---
title: "One hot encode categorical features"
author: "Jan Meppe"
date: 2021-04-09
description: "One hot encode categorical features"
type: technical_note
draft: false
---
## Imports


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
```

## Create  data


```python
d = {'fruit': ['apple', 'pear', 'apple', 'pear', 'pear']}
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
      <th>fruit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pear</td>
    </tr>
    <tr>
      <th>2</th>
      <td>apple</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pear</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pear</td>
    </tr>
  </tbody>
</table>
</div>



## Initialise


```python
one_hot = OneHotEncoder()
```

## Train


```python
one_hot.fit(df)
```




    OneHotEncoder(categorical_features=None, categories=None, drop=None,
                  dtype=<class 'numpy.float64'>, handle_unknown='error',
                  n_values=None, sparse=True)



## Apply


```python
one_hot.transform(df)
```




    <5x2 sparse matrix of type '<class 'numpy.float64'>'
    	with 5 stored elements in Compressed Sparse Row format>



## View


```python
one_hot.transform(df).toarray()
```




    array([[1., 0.],
           [0., 1.],
           [1., 0.],
           [0., 1.],
           [0., 1.]])



## Option 2 (`pd.get_dummies`)


```python
pd.get_dummies(df['fruit'])
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
      <th>apple</th>
      <th>pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


