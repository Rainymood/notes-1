---
title: "Create count items in list"
author: "Jan Meppe"
date: 2021-04-14
description: "Create count items in list"
type: technical_note
draft: false
---
## Imports


```python
from collections import Counter
```

## Create count


```python
# Create count for items in list
counter = Counter(['orange', 'banana', 'orange', 'fruit'])

# View counter
counter
```




    Counter({'orange': 2, 'banana': 1, 'fruit': 1})


