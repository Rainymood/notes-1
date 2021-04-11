---
title: "Create simulated data for clustering"
author: "Jan Meppe"
date: 2021-04-09
description: "Create simulated data for clustering"
type: technical_note
draft: false
---
# Create simulated data for clustering (blobs)

## Imports


```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np 

```

## Create data


```python
# Make features (data) and label (labels)
data, labels = make_blobs(n_samples = 200,
                          n_features = 2, 
                          centers = 3, # 3 clusters
                          cluster_std = 0.5, 
                          shuffle =True)
```

## Set up colours


```python
from itertools import islice, cycle

color_list = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
amount_of_labels = int(max(labels) + 1)
colors = np.array(list(islice(cycle(color_list), amount_of_labels)))
```

## Plot data


```python
x = data[:, 0]
y = data[:, 1]

plt.scatter(x, y, c=colors[labels])
plt.show()
```


    
![png](Create_simulated_data_for_clustering_%28blobs%29_files/Create_simulated_data_for_clustering_%28blobs%29_9_0.png)
    

