---
title: "Get n'th row of dataframe"
author: "Jan Meppe"
date: 2021-04-11
description: "Get n'th row of dataframe"
type: technical_note
draft: false
---
## Imports


```python
# Load library
import pandas as pd
```

## Create data


```python
df = pd.DataFrame({
    "apples": [1, 1],
    "oranges": [2, 4]
})
```

## Get dataframe with `.iloc[[]]`


```python
df.iloc[[0]]
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
      <th>apples</th>
      <th>oranges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## Get series with `iloc[]`


```python
df.iloc[0]
```




    apples     1
    oranges    2
    Name: 0, dtype: int64


