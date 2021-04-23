---
title: "Image"
author: "Jan Meppe"
date: 2021-04-14
description: "Image"
type: technical_note
draft: false
---

To show images in react native, use the `Image` component.

## Option 1: use `require`

You can use `require` to load in an image. This *can not* be done dynamically. The link must be known at build time.

```js
<Image
  style={styles.tinyLogo}
  source={require('@expo/snack-static/react-native-logo.png')}
/>
```

## Option 2: download from network

You can use the `uri` parameter with a URL to download images from a network.

Note that we pass in an object with a `uri` key.

```js
<Image
  style={styles.tinyLogo}
  source={{
    uri: 'https://reactnative.dev/img/tiny_logo.png',
  }}
/>
```

## Option 3: uri

You can load images from the uri. 

Note that we pass in an object with a `uri` key.

```js
<Image
  style={styles.logo}
  source={{
    uri: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAzCAYAAAA6oTAqAAAAEXRFWHRTb2Z0d2FyZQBwbmdjcnVzaEB1SfMAAABQSURBVGje7dSxCQBACARB+2/ab8BEeQNhFi6WSYzYLYudDQYGBgYGBgYGBgYGBgYGBgZmcvDqYGBgmhivGQYGBgYGBgYGBgYGBgYGBgbmQw+P/eMrC5UTVAAAAABJRU5ErkJggg==',
  }}
```

Read the official documentation [here](https://reactnative.dev/docs/image).