import { TextClassifier } from "../index";

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
