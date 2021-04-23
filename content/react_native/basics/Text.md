---
title: "Text"
author: "Jan Meppe"
date: 2021-04-14
description: "Text"
type: technical_note
draft: false
---

Another [fundamental component](https://reactnative.dev/docs/components-and-apis#basic-components) in react native is the `Text` component. 

The `Text` component is a component for displaying text. 

The `Text` component supports nesting, styling, and touch handling. 

Read the official documentation by clicking [here](https://reactnative.dev/docs/text).

## Example

```js
import React, { useState } from "react";
import { Text, StyleSheet } from "react-native";

const TextInANest = () => {
  const [titleText, setTitleText] = useState("Bird's Nest");
  const bodyText = useState("This is not really a bird nest.");

  const onPressTitle = () => {
    setTitleText("Bird's Nest [pressed]");
  };

  return (
    <Text style={styles.baseText}>
      <Text style={styles.titleText} onPress={onPressTitle}>
        {titleText}
        {"\n"}
        {"\n"}
      </Text>
      <Text numberOfLines={5}>{bodyText}</Text>
    </Text>
  );
};

const styles = StyleSheet.create({
  baseText: {
    fontFamily: "Cochin"
  },
  titleText: {
    fontSize: 20,
    fontWeight: "bold"
  }
});

export default TextInANest;
```