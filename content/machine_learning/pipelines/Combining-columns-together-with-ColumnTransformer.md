---
title: "Combining columns together with ColumnTransformer"
author: "Jan Meppe"
date: 2021-04-09
description: "Combining columns together with ColumnTransformer"
type: technical_note
draft: false
---
# Combining columns together with ColumnTransformer

From: 
* Hands on Machine learning

## Imports


```python
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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

def get_numerical_columns(df): 
    return list(df.select_dtypes('number').columns)

non_numerics = get_non_numerical_columns(df)
numerics = get_numerical_columns(df)
```

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

## Create pipeline


```python
cat_pipeline = Pipeline([('cat_selector', ColumnSelector(non_numerics))])
num_pipeline = Pipeline([('num_selector', ColumnSelector(numerics))])

# Syntax: name, transformer, column
full_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, non_numerics),
    ('num', num_pipeline, numerics)
])
```

## Fit pipeline


```python
full_pipeline.fit(df)
```




    ColumnTransformer(n_jobs=None, remainder='drop', sparse_threshold=0.3,
                      transformer_weights=None,
                      transformers=[('cat',
                                     Pipeline(memory=None,
                                              steps=[('cat_selector',
                                                      ColumnSelector(columns=['label']))],
                                              verbose=False),
                                     ['label']),
                                    ('num',
                                     Pipeline(memory=None,
                                              steps=[('num_selector',
                                                      ColumnSelector(columns=['score']))],
                                              verbose=False),
                                     ['score'])],
                      verbose=False)



## Transform pipeline


```python
full_pipeline.transform(df)
```




    array([['dog', 1],
           ['cat', 2],
           ['catdog', 3],
           ['dog', 4],
           ['catdog', 5]], dtype=object)


