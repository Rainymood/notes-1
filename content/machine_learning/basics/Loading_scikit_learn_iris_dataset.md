---
title: "Loading scikit learn iris dataset"
author: "Jan Meppe"
date: 2021-04-09
description: "Loading scikit learn iris dataset."
type: technical_note
draft: false
---
# Loading scikit-learn Iris dataset


```python
from sklearn import datasets
import numpy as np
```

## Load Iris Dataset

The [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) is one of the most famous databases for classification.

The dataset contains: 

* 3 classes (species of flowers)
* 50 observations per class


```python
# Load Iris datset
iris = datasets.load_iris()

# Create features
X = iris.data

# Create label
y = iris.target

# View first row
X[0]
```




    array([5.1, 3.5, 1.4, 0.2])



## Option 2: Load as frame


```python
# np.c_ is the numpy concatenate function
# which is used to concat iris['data'] and iris['target'] arrays 
# for pandas column argument: concat iris['feature_names'] list
# and string list (in this case one string); you can make this anything you'd like..  
# the original dataset would probably call this ['Species']
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])

df.head()
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
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Load as frame (from 0.23)

An even better way is to set `as_frame` to `True` which loads it as a pandas DataFrame instead of an array.


```python
import sklearn
sklearn.__version__
```




    '0.21.1'




```python
# iris = datasets.load_iris(as_frame=True)
```
