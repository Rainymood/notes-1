---
title: "Impute missing data"
author: "Jan Meppe"
date: 2021-04-09
description: "Impute missing data"
type: technical_note
draft: false
---
## Imports


```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
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



## Initialise imputer


```python
imputer = SimpleImputer(strategy="mean")
```

## Train imputer


```python
# Fit the imputer 
imputer = imputer.fit(df)
```

## Apply imputer


```python
# Apply the imputer
df_imputed = imputer.transform(df)

# View data
df_imputed
```




    array([[ 1.        ,  3.        ],
           [ 8.33333333,  4.        ],
           [10.        ,  5.        ],
           [14.        ,  4.        ]])


