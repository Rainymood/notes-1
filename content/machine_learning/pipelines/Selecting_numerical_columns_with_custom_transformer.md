---
title: "Selecting numerical columns with ColumnSelector"
author: "Jan Meppe"
date: 2021-04-09
description: "Selecting numerical columns with ColumnSelector"
type: technical_note
draft: false
---
# Selecting columns with a custom transformer (ColumnSelector)

From: 
* https://towardsdatascience.com/pipeline-columntransformer-and-featureunion-explained-f5491f815f

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
numerical = list(df.select_dtypes('number').columns)
numerical
```




    ['score']



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
num_pipeline = Pipeline([('num_selector', ColumnSelector(numerical))])
```

## Fit pipeline


```python
num_pipeline.fit(df)
```




    Pipeline(memory=None,
             steps=[('num_selector', ColumnSelector(columns=['score']))],
             verbose=False)



## Transform pipeline


```python
num_pipeline.transform(df)
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
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


