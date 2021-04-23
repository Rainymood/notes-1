---
title: "Persist AsyncStorage"
author: "Jan Meppe"
date: 2021-04-14
description: "Persist AsyncStorage"
type: technical_note
draft: false
---



```js
import AppLoading from 'expo-app-loading';
import AsyncStorage from '@react-native-async-storage/async-storage';
import App from './components/App';

export default function App(): {
  const [state, setState] = useState(null);

  const loadData = async () => {
    const storedState = await AsyncStorage.getItem("state");
    setState(storedState);
  }
    
  useEffect(() => {
    loadData();
  }, []);
  
  if (!state) {
    return (<AppLoading />); 
  }
    
  return (
    <App state={state} />
  );
}
```