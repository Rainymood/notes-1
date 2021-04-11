---
title: "Discretize features"
author: "Jan Meppe"
date: 2021-04-09
description: "Discretize features"
type: technical_note
draft: false
---
## Imports


```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import Binarizer
```

## Create  data


```python
d = {'values': [6, 10, 12, 100]}
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
      <th>values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



## Option 1: Binarize into 0/1


```python
binarizer = Binarizer(10)
```


```python
binarizer.fit_transform(df)
```




    array([[0],
           [0],
           [1],
           [1]])



## Option 2: Break feature into bins


```python
np.digitize(df, bins=[10, 15])
```




    array([[0],
           [1],
           [1],
           [2]])


