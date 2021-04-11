---
title: "Simple Boston pipeline"
author: "Jan Meppe"
date: 2021-04-09
description: "Simple Boston pipeline"
type: technical_note
draft: false
---
## Imports


```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
```

## Load in data


```python
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'])
```

## Create pipeline


```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', PCA()),
    ('regressor', Ridge())
])
```

## Fit pipeline


```python
pipe = pipe.fit(X_train, y_train)
```

## View parameters with `get_params()`


```python
pipe.get_params()
```




    {'memory': None,
     'steps': [('scaler',
       StandardScaler(copy=True, with_mean=True, with_std=True)),
      ('reduce_dim',
       PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
           svd_solver='auto', tol=0.0, whiten=False)),
      ('regressor',
       Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
             normalize=False, random_state=None, solver='auto', tol=0.001))],
     'verbose': False,
     'scaler': StandardScaler(copy=True, with_mean=True, with_std=True),
     'reduce_dim': PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
         svd_solver='auto', tol=0.0, whiten=False),
     'regressor': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
           normalize=False, random_state=None, solver='auto', tol=0.001),
     'scaler__copy': True,
     'scaler__with_mean': True,
     'scaler__with_std': True,
     'reduce_dim__copy': True,
     'reduce_dim__iterated_power': 'auto',
     'reduce_dim__n_components': None,
     'reduce_dim__random_state': None,
     'reduce_dim__svd_solver': 'auto',
     'reduce_dim__tol': 0.0,
     'reduce_dim__whiten': False,
     'regressor__alpha': 1.0,
     'regressor__copy_X': True,
     'regressor__fit_intercept': True,
     'regressor__max_iter': None,
     'regressor__normalize': False,
     'regressor__random_state': None,
     'regressor__solver': 'auto',
     'regressor__tol': 0.001}




```python
print('Testing score: ', pipe.score(X_test, y_test))
```

    Testing score:  0.7391809400706117


## Fine tune model with `GridSearchCV`


```python
n_components = np.arange(1, 11)
alpha = 2.0**np.arange(-6, 6)

params = {
    'reduce_dim__n_components': n_components,
    'regressor__alpha': alpha
}

gridsearch = GridSearchCV(pipe, params, verbose=1).fit(X_train, y_train)
```

    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
      warnings.warn(CV_WARNING, FutureWarning)
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    Fitting 3 folds for each of 120 candidates, totalling 360 fits


    [Parallel(n_jobs=1)]: Done 360 out of 360 | elapsed:    1.3s finished
    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/model_selection/_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)



```python
gridsearch.score(X_test, y_test)
```




    0.7004888795698064




```python
gridsearch.best_params_
```




    {'reduce_dim__n_components': 6, 'regressor__alpha': 2.0}


