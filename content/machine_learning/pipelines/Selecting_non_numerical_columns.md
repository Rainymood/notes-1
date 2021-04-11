---
title: "Selecting non-numerical columns"
author: "Jan Meppe"
date: 2021-04-09
description: "Selecting non-numerical columns"
type: technical_note
draft: false
---
## Imports


```python
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
```

## Create  data


```python
data = {'label': ['dog', 'cat', 'catdog', 'dog', 'catdog'], 'score': [1, 2, 3, 4, 5]}
df = pd.DataFrame(data, columns = ["label", "score"])
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
      <th>label</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>dog</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>catdog</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dog</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>catdog</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## Define numerical columns


```python
def get_non_numerical_columns(df):
    numerics = list(df.select_dtypes('number').columns)
    cols = list(df.columns)
    return [x for x in cols if x not in numerics]

non_numerics = get_non_numerical_columns(df)
print(non_numerics)
```

    ['label']


## Create custom transformer (fit and transform methods)


```python
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]
```

## Create numerical pipeline


```python
cat_pipeline = Pipeline([('cat_selector', ColumnSelector(non_numerics))])
```

## Fit pipeline


```python
cat_pipeline.fit(df)
```




    Pipeline(memory=None,
             steps=[('cat_selector', ColumnSelector(columns=['label']))],
             verbose=False)



## Transform pipeline


```python
cat_pipeline.transform(df)
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
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>dog</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>catdog</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dog</td>
    </tr>
    <tr>
      <th>4</th>
      <td>catdog</td>
    </tr>
  </tbody>
</table>
</div>



## From

* https://towardsdatascience.com/pipeline-columntransformer-and-featureunion-explained-f5491f815f
