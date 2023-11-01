# Dogma Project Machine Learning

**Dogma Machine Learning** is a ML framework in an active development. Currently finished only text classification module.

## Installation

```
git clone git@github.com:Dogma-Project/machine-learning.git
```

or

```
npm i dogma-ml
```

## Usage

### Runtime demo

```
npm run example
```

### Example

```
import { TextClassifier } from "dogma-ml";

async function run() {
  const dataset = [
    { input: "Hello world!", output: 0 },
    { input: "Hi there!", output: 0 },
    { input: "Hi", output: 0 },
    { input: "Hello my friend!", output: 0 },
    { input: "Hey, hi", output: 0 },
    { input: "Bye bye!", output: 1 },
    { input: "Bye, no hello", output: 1 }, // edit
    { input: "See you later!", output: 1 },
    { input: "Goodbye dude!", output: 1 },
    { input: "Good bye", output: 1 },
  ];
  const classifier = new TextClassifier({});
  const path = "data/model.json";
  try {
    await classifier.loadModel(path);
    console.log("Model loaded");
  } catch (err) {
    // err
  }
  const _res = await classifier.train(dataset);
  console.log("NOT PREDICTED:", _res.notPredicted);
  await classifier.saveModel(path);
  const res1 = classifier.predict("Hi, my dear friend!");
  const res2 = classifier.predict("Bye, see you later!");
  console.log(res1);
  console.log(res2);
}

run();
```

## History

TODO: Write history

## Credits

TODO: Write credits

## License

MIT License

Copyright (c) 2023 Dogma Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
