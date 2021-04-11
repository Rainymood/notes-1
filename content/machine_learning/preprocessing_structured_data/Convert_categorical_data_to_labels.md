---
title: "Convert categorical data to labels"
author: "Jan Meppe"
date: 2021-04-09
description: "Convert categorical data to labels"
type: technical_note
draft: false
---
`LabelEncoder` is just another scikit-learn estimator with a `fit()` method and a `transform()` method.

## Imports


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
```

## Create  data


```python
data = {'label': ['dog', 'cat', 'catdog', 'dog', 'catdog']}
df = pd.DataFrame(data, columns = ["label"])
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



## Initialise LabelEncoder


```python
le = LabelEncoder()
```

## Fit LabelEncoder


```python
le.fit(df['label'])
```




    LabelEncoder()



## View labels


```python
list(le.classes_)
```




    ['cat', 'catdog', 'dog']



## Transform label into number


```python
le.transform(df['label'])
```




    array([2, 0, 1, 2, 1])



## Transform number back into label


```python
list(le.inverse_transform([0, 1, 2]))
```




    ['cat', 'catdog', 'dog']



## Source

* [chrisalbon.com](https://chrisalbon.com/machine_learning/preprocessing_structured_data/convert_pandas_categorical_column_into_integers_for_scikit-learn/)
