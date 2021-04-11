---
title: "Iris pipeline"
author: "Jan Meppe"
date: 2021-04-11
description: "Iris pipeline"
type: technical_note
draft: false
---
## Imports


```python
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
```

## Load data


```python
iris = datasets.load_iris()
```


```python
df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)

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



## Visualise data and inspect for missing values/outliers

* First thing we want to do is inspect the data to see if we have any missing values or outliers


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
    sepal length (cm)    150 non-null float64
    sepal width (cm)     150 non-null float64
    petal length (cm)    150 non-null float64
    petal width (cm)     150 non-null float64
    target               150 non-null float64
    dtypes: float64(5)
    memory usage: 5.9 KB


* `df.info()` shows that there are no missing values as we have 150 non-null values


```python
sns.pairplot(df)
```




    <seaborn.axisgrid.PairGrid at 0x11e375358>




    
![png](Iris_pipeline_files/Iris_pipeline_10_1.png)
    


* `sns.pairplot` gives a beautiful overview of the data

## Train-test split


```python
df_ = df.copy()
y = df_.pop('target')
X = df_
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

## Setup pipeline


```python
pipe = Pipeline([
    ("std", StandardScaler()),
    ("clf", SVC(kernel="linear", C=1))
])
```

## Train pipeline


```python
pipe.fit(X_train, y_train)
```




    Pipeline(memory=None,
             steps=[('std',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('clf',
                     SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
                         decision_function_shape='ovr', degree=3,
                         gamma='auto_deprecated', kernel='linear', max_iter=-1,
                         probability=False, random_state=None, shrinking=True,
                         tol=0.001, verbose=False))],
             verbose=False)



## Training set accuracy


```python
pipe.score(X_train, y_train)
```




    0.96



## Test set accuracy


```python
pipe.score(X_test, y_test)
```




    0.98



## Grid search

First check the parameters of the pipeline.


```python
pipe.get_params()
```




    {'memory': None,
     'steps': [('std', StandardScaler(copy=True, with_mean=True, with_std=True)),
      ('clf', SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
           decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
           kernel='linear', max_iter=-1, probability=False, random_state=None,
           shrinking=True, tol=0.001, verbose=False))],
     'verbose': False,
     'std': StandardScaler(copy=True, with_mean=True, with_std=True),
     'clf': SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
         decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
         kernel='linear', max_iter=-1, probability=False, random_state=None,
         shrinking=True, tol=0.001, verbose=False),
     'std__copy': True,
     'std__with_mean': True,
     'std__with_std': True,
     'clf__C': 1,
     'clf__cache_size': 200,
     'clf__class_weight': None,
     'clf__coef0': 0.0,
     'clf__decision_function_shape': 'ovr',
     'clf__degree': 3,
     'clf__gamma': 'auto_deprecated',
     'clf__kernel': 'linear',
     'clf__max_iter': -1,
     'clf__probability': False,
     'clf__random_state': None,
     'clf__shrinking': True,
     'clf__tol': 0.001,
     'clf__verbose': False}



We want to try different values for `C = (0.01, 0.10, 1.0, 10, 100)` so that is `clf__C`


```python
params = {
    "clf__C": [0.01, 0.1, 1.0, 10, 100]
}
```


```python
gridsearch = GridSearchCV(estimator=pipe, param_grid=params, n_jobs=-1)
gridsearch.fit(X_train, y_train)
```

    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
      warnings.warn(CV_WARNING, FutureWarning)
    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/model_selection/_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)





    GridSearchCV(cv='warn', error_score='raise-deprecating',
                 estimator=Pipeline(memory=None,
                                    steps=[('std',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('clf',
                                            SVC(C=1, cache_size=200,
                                                class_weight=None, coef0=0.0,
                                                decision_function_shape='ovr',
                                                degree=3, gamma='auto_deprecated',
                                                kernel='linear', max_iter=-1,
                                                probability=False,
                                                random_state=None, shrinking=True,
                                                tol=0.001, verbose=False))],
                                    verbose=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'clf__C': [0.01, 0.1, 1.0, 10, 100]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
print('Best score for train set:', gridsearch.best_score_) 
```

    Best score for train set: 0.94



```python
print('Best param C:', gridsearch.best_params_['clf__C'])
```

    Best param C: 0.1



```python
print("Best score for test set:", gridsearch.score(X_test, y_test))
```

    Best score for test set: 1.0



```python

```
