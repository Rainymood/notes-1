---
title: "View"
author: "Jan Meppe"
date: 2021-04-14
description: "View"
type: technical_note
draft: false
---

One of the most fundamental components in react native is the `View` component.

Read the official documentation [here](https://reactnative.dev/docs/view). 

The `View` component is analogous to the `div` component. It can be thought
of as a container. More specifically the `View` component is just a container
that supports `flexbox`, `styling`, and some more things that are less
important.

`View`s are designed to be nested and that is exactly how you will build your
react native application, by creating different views with other components
inside.

## Example

```js
import React from "react";
import { View, Text } from "react-native";

const ViewBoxesWithColorAndText = () => {
  return (
    <View
      style={{
        flexDirection: "row",
        height: 100,
        padding: 20
      }}
    >
      <View style={{ backgroundColor: "blue", flex: 0.3 }} />
      <View style={{ backgroundColor: "red", flex: 0.5 }} />
      <Text>Hello World!</Text>
    </View>
  );
};

export default ViewBoxesWithColorAndText;
```