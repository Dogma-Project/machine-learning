const { TextClassifier } = require("./index");

async function run() {
  const dataset = [
    { input: "Hello world!", output: 0 },
    { input: "Hi there!", output: 0 },
    { input: "Hello my friend!", output: 0 },
    { input: "Hey, hi", output: 0 },
    { input: "Bye!", output: 1 },
    { input: "See you later!", output: 1 },
    { input: "Goodbye!", output: 1 },
    { input: "Good bye", output: 1 },
  ];
  const classifier = new TextClassifier({
    learningAccuracy: 1.2,
    trainingThreshold: 0.95,
  });
  const path = "./data/model.json";
  classifier
    .train(dataset)
    .then((_res) => {
      return classifier.saveModel(path);
    })
    .then(() => {
      const res1 = classifier.predict("Hi, my dear friend!");
      const res2 = classifier.predict("Bye, see you later!");
      console.log(res1, res2);
    });
}

run();
