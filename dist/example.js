"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
const index_1 = require("./index");
function run() {
    return __awaiter(this, void 0, void 0, function* () {
        const dataset = [
            { input: "Hello world!", output: 0 },
            { input: "Hi there!", output: 0 },
            { input: "Hi", output: 0 },
            { input: "Hello my friend!", output: 0 },
            { input: "Hey, hi", output: 0 },
            { input: "Bye bye!", output: 1 },
            { input: "Bye, no hello", output: 1 },
            { input: "See you later!", output: 1 },
            { input: "Goodbye dude!", output: 1 },
            { input: "Good bye", output: 1 },
        ];
        const classifier = new index_1.TextClassifier({});
        const path = "data/model.json";
        classifier
            .train(dataset)
            .then((_res) => {
            console.log("NOT PREDICTED:", _res.notPredicted);
            return classifier.saveModel(path);
        })
            .then(() => {
            const res1 = classifier.predict("Hi, my dear friend!");
            const res2 = classifier.predict("Bye, see you later!");
            console.log(res1, res2);
        });
        /*
        classifier.loadModel(path).then(() => {
          const res1 = classifier.predict("Hi, my dear friend!");
          const res2 = classifier.predict("Bye, see you later!");
          console.log(res1, res2);
        });
        */
    });
}
run();
