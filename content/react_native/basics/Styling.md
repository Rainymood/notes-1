---
title: "Styling"
author: "Jan Meppe"
date: 2021-04-14
description: "Styling"
type: technical_note
draft: false
---

There are two ways of adding styling to a component: 

1. Inline styles
2. Stylesheets

## Option 1: Inline

Add the style inline. This is quick and dirty. I actually prefer to code this way and then refactor it later.

```js
import React from "react";
import { StyleSheet, Text, View } from "react-native";

const App = () => (
  <View style={{
    flex: 1,
    padding: 24,
    backgroundColor: "#eaeaea"
  }}>
    <Text>React Native</Text>
  </View>
);
```

## Option 2: Stylesheets

Create a `StyleSheet` object and put the styles in there.

```js
import React from "react";
import { StyleSheet, Text, View } from "react-native";

const App = () => (
  <View style={styles.container}>
    <Text>React Native</Text>
  </View>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 24,
    backgroundColor: "#eaeaea"
  }
});

```