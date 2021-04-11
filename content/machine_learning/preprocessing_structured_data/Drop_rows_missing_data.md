---
title: "Drop rows missing data"
author: "Jan Meppe"
date: 2021-04-09
description: "Drop rows missing data"
type: technical_note
draft: false
---
## Imports


```python
import pandas as pd
import numpy as np
```

## Create  data


```python
d = {'col1': [1, np.nan, 10, 14], 'col2': [3, 4, 5, np.nan]}
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Dropping rows with `dropna()`


```python
df.dropna()
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



## Deleting missing values with `isnull()`


```python
mask = df.isnull().any(axis=1) # np.isnan(df) also works
df[~mask]
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>


