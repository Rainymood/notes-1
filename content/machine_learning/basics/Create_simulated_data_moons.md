---
title: "Create simulated data moons"
author: "Jan Meppe"
date: 2021-04-09
description: "Create simulated data moons"
type: technical_note
draft: false
---
# Create simulate data moons

## Imports 


```python
from itertools import cycle, islice
from sklearn.datasets import make_moons 
```

## Create data

Generate `X` and `y` where `X` is the position and `label` is the label. 


```python
X, labels = make_moons(noise=0.1) 
```

## Generate colormap (what the fuck?)


```python
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                                     int(max(label) + 1))))
```

## View data


```python
import matplotlib.pyplot as plt


x = X[:,0]
y = X[:,1]

plt.scatter(x, y, color=colors[labels])
plt.show()
```


    
![png](Create_simulated_data_moons_9_0.png)
    

