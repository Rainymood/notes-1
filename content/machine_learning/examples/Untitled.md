```python
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
```


```python
# Load Iris datset
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




    <seaborn.axisgrid.PairGrid at 0x11c98c908>




    
![png](Untitled_files/Untitled_6_1.png)
    


* `sns.pairplot` gives a beautiful overview of the data


```python

```


```python
df_ = df.copy()
y = df_.pop('target')
X = df_
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```


```python
pipe = Pipeline([
    ("std", StandardScaler()),
    ("clf", SVC(kernel="linear", C=1))
])
```


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




```python
pipe.score(X_train, y_train)
```




    0.97




```python
pipe.score(X_test, y_test)
```




    0.98




```python
X_test.iloc[[10]]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86</th>
      <td>6.7</td>
      <td>3.1</td>
      <td>4.7</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
