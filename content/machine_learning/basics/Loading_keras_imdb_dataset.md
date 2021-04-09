---
title: "Loading keras imdb dataset"
author: "Jan Meppe"
date: 2021-04-09
description: "Loading keras imdb dataset"
type: technical_note
draft: false
---
# Loading Keras IMDB dataset


```python
import numpy as np
import pandas as pd

from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
```

This downloads 25.000 movie reviews from IMDB with the label positive/negative.

Each review is encoded as a list of indexes (integers)

Words are encoded by overall frequency in the dataset. Integer 3 encodes the 3rd most frequent word in the data. 

## Create data


```python
# set number of features (use top 1000 words)
num_words = 1000
index_from = 3

# load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words, index_from=index_from)
```

## Convert word to index (and reverse)

Note that by default indexing starts from 3 in the default settings as follows.

```
start_char=1, oov_char=2, index_from=3,
```

This means that: 

* `0` is the padding char
* `1` is the start char
* `2` is oov
* `3` is unknown

So we need to load in the mapping and then shift it 3 to the right, very convenient.


```python
# Get word index (word => index)
word_to_id = imdb.get_word_index()
word_to_id = {k:(v+index_from) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
word_to_id["<UNUSED>"] = 3

# Reverse word index (index => word)
id_to_word = dict([(value, key) for (key, value) in word_to_id.items()])
```

## View data


```python
" ".join([id_to_word[i] for i in x_train[1000]])
```




    "<START> although i had seen <UNK> in a theater way back in <UNK> i couldn't remember anything of the plot except for <UNK> <UNK> of <UNK> <UNK> running and fighting against a <UNK> of <UNK> <UNK> and <UNK> <UNK> the ending br br after reading some of the other reviews i <UNK> up a <UNK> of the <UNK> released dvd to once again <UNK> the world of <UNK> br br it turns out this is one of those films <UNK> during the <UNK> that would go <UNK> to video today the film stars <UNK> <UNK> <UNK> <UNK> as <UNK> <UNK> <UNK> out of the <UNK> to <UNK> the <UNK> of <UNK> to <UNK> and <UNK> <UNK> the game a <UNK> <UNK> <UNK> by the <UNK> who <UNK> his people by <UNK> what sounds like <UNK> power the <UNK> of the <UNK> <UNK> the star <UNK> <UNK> <UNK> <UNK> is <UNK> in the <UNK> <UNK> by <UNK> <UNK> who never <UNK> or leaves the house once <UNK> tries to <UNK> in with the <UNK> by <UNK> a <UNK> red <UNK> with <UNK> of <UNK> and white <UNK> to say <UNK> finds himself running and fighting for his life along the <UNK> <UNK> of <UNK> on his way to a <UNK> with <UNK> and the game br br star <UNK> <UNK> was <UNK> <UNK> by director robert <UNK> who it looks like was never on the set the so called script is just this side of <UNK> see other reviews for the many <UNK> throughout the town of <UNK> has a few good moments but is <UNK> <UNK> by bad editing the ending <UNK> still there's the <UNK> of a good action <UNK> here a <UNK> <UNK> version with more <UNK> action and <UNK> <UNK> might even be pretty good"



## Tokenize


```python
# Convert list of integers to one-hot matrix
tokenizer = Tokenizer(num_words=num_words)
```


```python
# Convert list of integers into matrix with max length of dict
train_features = tokenizer.sequences_to_matrix(x_train, mode='binary')
test_features = tokenizer.sequences_to_matrix(x_test, mode='binary')

print(f"train features shape: {train_features.shape}, test features shape {test_features.shape}") 
```

    train features shape: (25000, 1000), test features shape (25000, 1000)


Now you have: 

* `(25000, 1000)` train features ohe matrix
* `(25000, 1000)` test features ohe matrix 
* `(25000,)` train labels
* `(25000,)` test labels
