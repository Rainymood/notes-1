---
title: "Create simulated data for regression"
author: "Jan Meppe"
date: 2021-04-09
description: "Create simulated data for regression"
type: technical_note
draft: false
---
# Creating simulated data for regression

Generate a problem set for regression with `make_regression`.

## Imports


```python
import pandas as pd
from sklearn.datasets import make_regression
```

## Create data


```python
n_samples = 100
n_features = 1
X, y, coeff = make_regression(n_samples=n_samples,
                              n_features=n_features,
                              noise = 10,
                              coef=True)
```

## View data


```python
pd.DataFrame(X).head()
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.308898</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.116417</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.844613</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.022316</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.541214</td>
    </tr>
  </tbody>
</table>
</div>




```python
X.shape
```




    (100, 1)




```python
y.shape
```




    (100,)



## Plot data


```python
import matplotlib.pyplot as plt

plt.plot(X, y, 'bo')
plt.show()
```


    
![png](Create_simulated_data_for_regression_11_0.png)
    

