---
title: "Should you train preprocessing on the test set?"
author: "Jan Meppe"
date: 2021-04-09
description: "Should you train preprocessing on the test set?"
type: technical_note
draft: false
---

**NO. NEVER PREPROCESS YOUR TEST SET**

This is a mistake because it leaks data from your train set into your test set.

Consider [this example](https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines), first a processing routine is applied:

```python
def processing(df):
    ...
    return(df)

df = processing(df)
```

And then later the data is split into a test and train set:

```python
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33, random_state=42)
```

This is wrong, and only right by accident.

Do this the other way around. First split, then train your preprocessing on the train set. 

See [this answer](https://stats.stackexchange.com/questions/321559/pre-processing-applied-on-all-three-training-validation-test-sets):

"You should do the same preprocessing on all your data however if that preprocessing depends on the data (e.g. standardization, pca) then you should calculate it on your training data and then use the parameters from that calculation to apply it to your validation and test data.

For example if you are centering your data (subtracting the mean) then you should calculate the mean on your training data ONLY and then subtract that same mean from all your data (i.e. subtract the mean of the training data from the validation and test data, DO NOT calculate 3 separate means)."