Idea parking space:

* Extract features from URL



What I've done so far: 

* Load in the data
* Inspect the data
* Look for unique values with (`df.value_counts`)
* Look for missing values with `df.info`
* Drop na values (todo: refactor this in pipeline)
* Create targets and remove from df

# Bot or not v2

This is version 2 of the bot or not framework where we try to incorporate more features and try to put everything in a single pipeline.


```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
```

## Load in the data


```python
filename = "../../../data/bot-or-not-clickdata.csv"
df = pd.read_csv(filename)
```

Data: 

* `epoch_ms`
* `session_id`
* `country_by_ip_address`
* `region_by_ip_address`
* `url_without_parameters`
* `referrer_without_parameters`
* `visitor_recognition_type`
* `ua_agent_class`



```python
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
      <th>epoch_ms</th>
      <th>session_id</th>
      <th>country_by_ip_address</th>
      <th>region_by_ip_address</th>
      <th>url_without_parameters</th>
      <th>referrer_without_parameters</th>
      <th>visitor_recognition_type</th>
      <th>ua_agent_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1520280001034</td>
      <td>be73c8d1b836170a21529a1b23140f8e</td>
      <td>US</td>
      <td>CA</td>
      <td>https://www.bol.com/nl/l/nederlandstalige-kuns...</td>
      <td>NaN</td>
      <td>ANONYMOUS</td>
      <td>Robot</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1520280001590</td>
      <td>c24c6637ed7dcbe19ad64056184212a7</td>
      <td>US</td>
      <td>CA</td>
      <td>https://www.bol.com/nl/l/italiaans-natuur-wete...</td>
      <td>NaN</td>
      <td>ANONYMOUS</td>
      <td>Robot</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1520280002397</td>
      <td>ee391655f5680a7bfae0019450aed396</td>
      <td>IT</td>
      <td>LI</td>
      <td>https://www.bol.com/nl/p/nespresso-magimix-ini...</td>
      <td>https://www.bol.com/nl/p/nespresso-magimix-ini...</td>
      <td>ANONYMOUS</td>
      <td>Browser</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1520280002598</td>
      <td>f8c8a696dd37ca88233b2df096afa97f</td>
      <td>US</td>
      <td>CA</td>
      <td>https://www.bol.com/nl/l/nieuwe-engelstalige-o...</td>
      <td>NaN</td>
      <td>ANONYMOUS</td>
      <td>Robot</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1520280004428</td>
      <td>f8b0c06747b7dd1d53c0932306bd04d6</td>
      <td>US</td>
      <td>CA</td>
      <td>https://www.bol.com/nl/l/nieuwe-actie-avontuur...</td>
      <td>NaN</td>
      <td>ANONYMOUS</td>
      <td>Robot Mobile</td>
    </tr>
  </tbody>
</table>
</div>



# Preprocessing

## Drop NaNs


```python
mask = df['region_by_ip_address'].isnull()
df = df.loc[~mask]
```

Let's check for missing data with `df.info()`


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 49886 entries, 0 to 59780
    Data columns (total 8 columns):
    epoch_ms                       49886 non-null int64
    session_id                     49886 non-null object
    country_by_ip_address          49886 non-null object
    region_by_ip_address           49886 non-null object
    url_without_parameters         49886 non-null object
    referrer_without_parameters    12838 non-null object
    visitor_recognition_type       49886 non-null object
    ua_agent_class                 49886 non-null object
    dtypes: int64(1), object(7)
    memory usage: 3.4+ MB


We have some missing values in: 
* `country`
* `region`
* `referrer_without_parameters`

First come up with a very simple model. 

* We drop the column `region_by_ip_address`
* We drop the column `referrer_without_parameters`

## Create target/labels

Let's check what categories we have:


```python
df['ua_agent_class'].value_counts()
```




    Browser              26667
    Robot                15852
    Robot Mobile          5115
    Browser Webview       1454
    Hacker                 690
    Special                102
    Mobile App               4
    Cloud Application        2
    Name: ua_agent_class, dtype: int64



We turn these into labels by picking the right ones and adding a zero or one there.


```python
def class_to_bot(agent):
    if agent in ["Robot", "Robot Mobile", "Special", "Cloud Application"]: 
        return 1
    else: 
        return 0
    
df['target'] = df['ua_agent_class'].apply(class_to_bot)

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
      <th>epoch_ms</th>
      <th>session_id</th>
      <th>country_by_ip_address</th>
      <th>region_by_ip_address</th>
      <th>url_without_parameters</th>
      <th>referrer_without_parameters</th>
      <th>visitor_recognition_type</th>
      <th>ua_agent_class</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1520280001034</td>
      <td>be73c8d1b836170a21529a1b23140f8e</td>
      <td>US</td>
      <td>CA</td>
      <td>https://www.bol.com/nl/l/nederlandstalige-kuns...</td>
      <td>NaN</td>
      <td>ANONYMOUS</td>
      <td>Robot</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1520280001590</td>
      <td>c24c6637ed7dcbe19ad64056184212a7</td>
      <td>US</td>
      <td>CA</td>
      <td>https://www.bol.com/nl/l/italiaans-natuur-wete...</td>
      <td>NaN</td>
      <td>ANONYMOUS</td>
      <td>Robot</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1520280002397</td>
      <td>ee391655f5680a7bfae0019450aed396</td>
      <td>IT</td>
      <td>LI</td>
      <td>https://www.bol.com/nl/p/nespresso-magimix-ini...</td>
      <td>https://www.bol.com/nl/p/nespresso-magimix-ini...</td>
      <td>ANONYMOUS</td>
      <td>Browser</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1520280002598</td>
      <td>f8c8a696dd37ca88233b2df096afa97f</td>
      <td>US</td>
      <td>CA</td>
      <td>https://www.bol.com/nl/l/nieuwe-engelstalige-o...</td>
      <td>NaN</td>
      <td>ANONYMOUS</td>
      <td>Robot</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1520280004428</td>
      <td>f8b0c06747b7dd1d53c0932306bd04d6</td>
      <td>US</td>
      <td>CA</td>
      <td>https://www.bol.com/nl/l/nieuwe-actie-avontuur...</td>
      <td>NaN</td>
      <td>ANONYMOUS</td>
      <td>Robot Mobile</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop(columns=['ua_agent_class'])
```

# Feature engineering




```python

```


```python

```


```python
y = df.pop('target')
X = df
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2656             try:
    -> 2657                 return self._engine.get_loc(key)
       2658             except KeyError:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'target'

    
    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-48-9fd451da30a0> in <module>
    ----> 1 y = df.pop('target')
          2 X = df


    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/pandas/core/generic.py in pop(self, item)
        807         3  monkey        NaN
        808         """
    --> 809         result = self[item]
        810         del self[item]
        811         try:


    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/pandas/core/frame.py in __getitem__(self, key)
       2925             if self.columns.nlevels > 1:
       2926                 return self._getitem_multilevel(key)
    -> 2927             indexer = self.columns.get_loc(key)
       2928             if is_integer(indexer):
       2929                 indexer = [indexer]


    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2657                 return self._engine.get_loc(key)
       2658             except KeyError:
    -> 2659                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2660         indexer = self.get_indexer([key], method=method, tolerance=tolerance)
       2661         if indexer.ndim > 1 or indexer.size > 1:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'target'



```python
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
      <th>epoch_ms</th>
      <th>session_id</th>
      <th>country_by_ip_address</th>
      <th>region_by_ip_address</th>
      <th>url_without_parameters</th>
      <th>referrer_without_parameters</th>
      <th>visitor_recognition_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1520280001034</td>
      <td>be73c8d1b836170a21529a1b23140f8e</td>
      <td>US</td>
      <td>CA</td>
      <td>https://www.bol.com/nl/l/nederlandstalige-kuns...</td>
      <td>NaN</td>
      <td>ANONYMOUS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1520280001590</td>
      <td>c24c6637ed7dcbe19ad64056184212a7</td>
      <td>US</td>
      <td>CA</td>
      <td>https://www.bol.com/nl/l/italiaans-natuur-wete...</td>
      <td>NaN</td>
      <td>ANONYMOUS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1520280002397</td>
      <td>ee391655f5680a7bfae0019450aed396</td>
      <td>IT</td>
      <td>LI</td>
      <td>https://www.bol.com/nl/p/nespresso-magimix-ini...</td>
      <td>https://www.bol.com/nl/p/nespresso-magimix-ini...</td>
      <td>ANONYMOUS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1520280002598</td>
      <td>f8c8a696dd37ca88233b2df096afa97f</td>
      <td>US</td>
      <td>CA</td>
      <td>https://www.bol.com/nl/l/nieuwe-engelstalige-o...</td>
      <td>NaN</td>
      <td>ANONYMOUS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1520280004428</td>
      <td>f8b0c06747b7dd1d53c0932306bd04d6</td>
      <td>US</td>
      <td>CA</td>
      <td>https://www.bol.com/nl/l/nieuwe-actie-avontuur...</td>
      <td>NaN</td>
      <td>ANONYMOUS</td>
    </tr>
  </tbody>
</table>
</div>




```python
class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return self
    
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        return self
```


```python
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Transformer that selects a particular feature."""

    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X[self.feature_names]
```


```python
class UrlLength(BaseEstimator, TransformerMixin):
    def __init__(self, url):
        self.url = url
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return len(X[self.url])
```


```python
from sklearn.pipeline import FeatureUnion
```


```python

```


```python

```


```python
url_pipeline = Pipeline([
    ('selector', FeatureSelector(['url_without_parameters'])),
    ("length", UrlLength('url_without_parameters'))
])

url_pipeline.fit_transform(X, y)
```




    49886




```python

```


```python

```


```python

```


```python
df = df.drop(columns=['epoch_ms', 'session_id', 'region_by_ip_address', 'referrer_without_parameters', 'url_without_parameters'])
```


```python
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
      <th>country_by_ip_address</th>
      <th>visitor_recognition_type</th>
      <th>ua_agent_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US</td>
      <td>ANONYMOUS</td>
      <td>Robot</td>
    </tr>
    <tr>
      <th>1</th>
      <td>US</td>
      <td>ANONYMOUS</td>
      <td>Robot</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IT</td>
      <td>ANONYMOUS</td>
      <td>Browser</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>ANONYMOUS</td>
      <td>Robot</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>ANONYMOUS</td>
      <td>Robot Mobile</td>
    </tr>
  </tbody>
</table>
</div>



# Prepare data for ML algorithm


```python
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30)
```


```python
pipe = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore')), 
    ('clf', RandomForestClassifier(n_estimators=10))
])
```


```python
pipe.fit(X_train, y_train)
```




    Pipeline(memory=None,
             steps=[('ohe',
                     OneHotEncoder(categorical_features=None, categories=None,
                                   drop=None, dtype=<class 'numpy.float64'>,
                                   handle_unknown='ignore', n_values=None,
                                   sparse=True)),
                    ('clf',
                     RandomForestClassifier(bootstrap=True, class_weight=None,
                                            criterion='gini', max_depth=None,
                                            max_features='auto',
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_impurity_split=None,
                                            min_samples_leaf=1, min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            n_estimators=10, n_jobs=None,
                                            oob_score=False, random_state=None,
                                            verbose=0, warm_start=False))],
             verbose=False)




```python
train_acc = pipe.score(X_train, y_train)
test_acc = pipe.score(X_test, y_test)
```


```python
print("Accuracy on train set:", train_acc)
print("Accuracy on test set:", test_acc)
```

    Accuracy on train set: 0.959077892325315
    Accuracy on test set: 0.9573032206334358



```python

```
