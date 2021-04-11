---
title: "Loading scikit-learn digits dataset"
author: "Jan Meppe"
date: 2021-04-09
description: "Loading scikit-learn digits dataset"
type: technical_note
draft: false
---
# Loading scikit-learn digits dataset


```python
from sklearn import datasets
import matplotlib.pyplot as plt
```

## Load digits dataset

The Digits dataset is a dataset of handwritten digits. Each feature is the intensity of a single pixel in an 8x8 image.


```python
# Load
digits = datasets.load_digits()
```


```python
# View attributes
dir(digits)
```




    ['DESCR', 'data', 'images', 'target', 'target_names']




```python
# Create features
X = digits.data

# Create target label
y = digits.target
```

Each of the futures are represented as a vector of length 64 instead of an 8x8 image. The `images` attribute is attached but we can also just reshape it ourselves. 


```python
X[0].reshape(8,8) == digits.images[0]
```




    array([[ True,  True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True,  True]])



## View data


```python
plt.matshow(digits.images[0]) # same: plt.matshow(X[0].reshape(8,8))
plt.show()
```


    
![png](Loading_scikit_learn_digits_dataset_files/Loading_scikit_learn_digits_dataset_10_0.png)
    


## Load in as dataframe (option 2)


```python
import numpy as np 
import pandas as pd

data = np.c_[digits['data'], digits['target']]
columns = list(range(64)) + ["target"]
df = pd.DataFrame(data=data, columns=columns)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
      <th>61</th>
      <th>62</th>
      <th>63</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>


