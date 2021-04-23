---
title: "Accuracy"
author: "Jan Meppe"
date: 2021-04-14
description: "Accuracy"
type: technical_note
draft: false
---
If you want to calculate the `accuracy` with cross-validation, check out [this](https://chrisalbon.com/machine_learning/model_evaluation/accuracy/). 

# Imports


```python
from sklearn.metrics import accuracy_score
```

# Create data


```python
y_true = [0, 1, 2, 3, 4]
y_pred = [0, 2, 1, 3, 5]
```

# Calculate accuracy


```python
accuracy_score(y_true, y_pred)
```




    0.4


