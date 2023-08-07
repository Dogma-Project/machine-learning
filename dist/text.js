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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const promises_1 = __importDefault(require("node:fs/promises"));
class TextClassifier {
    /**
     *
     * @param params
     * @param params.stemmer word stemming function
     * @param params.trainingThreshold
     * @param params.modelizeConstant
     * @param params.cleanReg regexp to clean text. default: /[^a-z0-9\ ']+/gi
     * @param params.median
     */
    constructor({ stemmer, trainingThreshold, modelizeConstant, cleanReg, median, }) {
        // configs
        this.stemmer = stemmer || this.pseudoStemmer;
        this.trainingThreshold = trainingThreshold || 0.99;
        this.modelizeConstant = modelizeConstant || 0.7;
        this.cleanReg = cleanReg || /[^a-z0-9\ ']+/gi;
        this.median = median || 0.05;
        this.voc = {};
        this.vocValues = [];
        this.dlrCache = {};
        this.model = {};
        this.outputs = [];
        this.initValue = 0.5;
        this.balance = [];
        this.ready = false;
        this.modelAccuracy = -1;
        this.accuracyRepeats = 0;
        this.accuracyRepeatsStopThreshold = 10;
        this.diffCache = {};
        this.maxWeight = 1;
        this.thresholds = {
            valueThreshold: null,
            betasThreshold: null,
        };
        this.notPredicted = [];
        this.predictedValues = [];
        this.predictedBetas = [];
    }
    /**
     *
     * @param input
     */
    pseudoStemmer(input) {
        return input.toLowerCase();
    }
    /**
     *
     * @param msg
     */
    tokenizeMessage(msg) {
        const arr = msg
            .replace(this.cleanReg, " ")
            .split(" ")
            .map((word) => {
            if (!word.length)
                return -1;
            return this.voc[this.stemmer(word)]
                ? this.voc[this.stemmer(word)].id
                : 0;
        });
        return arr.filter((token) => token !== -1);
    }
    /**
     *
     * @param tokenized
     */
    layerize(tokenized) {
        const result = tokenized.map((word, i, arr) => {
            if (i < tokenized.length - 1) {
                const val = `${word}:${arr[i + 1]}`;
                this.dlrCache[val] = [word, arr[i + 1]];
                return val;
            }
        });
        return result.filter((j) => j !== undefined);
    }
    getDiff(a, b) {
        const key = a + ":" + b;
        if (this.diffCache[key])
            return this.diffCache[key];
        const valA = this.vocValues[a];
        const valB = this.vocValues[b];
        let result = -1;
        let weight = 0;
        if (valA && valB) {
            if (valA.output === valB.output) {
                result = valA.output;
                weight = valA.value; // valA.value + valB.value
            }
            else if (valA.value > valB.value) {
                result = valA.output;
                weight = valA.value - valB.value;
            }
            else {
                result = valB.output;
                weight = valB.value - valA.value;
            }
        }
        else if (valA) {
            result = valA.output;
            weight = valA.value;
        }
        else if (valB) {
            result = valB.output;
            weight = valB.value;
        }
        const res = { result, weight };
        this.diffCache[key] = res;
        return res;
    }
    /**
     *
     * @param values
     * @param q
     * @param max default:true
     * @returns
     */
    getMedian(values, q, max = true) {
        if (max) {
            values.sort((a, b) => b - a);
        }
        else {
            values.sort();
        }
        values.splice(Math.ceil(values.length * q));
        return values.reduce((p, a) => p + a, 0) / values.length;
    }
    /**
     *
     * @param {Array} data {input, output}
     *
     */
    makeVocabulary(dataset) {
        console.time("voc");
        console.log("LOG:", "Making vocabulary...");
        this.voc = {};
        // set outputs
        const outputs = new Set();
        dataset.forEach((entry) => outputs.add(entry.output));
        this.outputs.forEach((entry) => outputs.add(entry));
        this.outputs = [...outputs];
        // prepare and count new data
        const temp = {};
        dataset.forEach((row) => {
            try {
                const arr = row.input.replace(this.cleanReg, " ").split(" ");
                if (!arr.length)
                    return;
                arr.forEach((word) => {
                    word = this.stemmer(word);
                    if (!word.length)
                        return;
                    if (!temp[word]) {
                        temp[word] = Array(this.outputs.length).fill(0);
                    }
                    const val = temp[word][row.output];
                    temp[word][row.output] = (val || 0) + 1;
                });
            }
            catch (err) {
                console.error(err);
            }
        });
        // add new entries to vocabulary
        for (let k of Object.keys(temp)) {
            if (!this.voc[k])
                this.voc[k] = {
                    id: -1,
                    output: 0,
                    value: 0,
                    stats: temp[k],
                };
        }
        // get new size
        const size = Object.keys(this.voc).length;
        // get size by outputs
        const sizes = [];
        Object.values(this.voc).forEach((item) => {
            const { stats } = item;
            for (let k = 0; k < stats.length; k++) {
                sizes[k] = (sizes[k] || 0) + (stats[k] || 0);
            }
        });
        let maxId = 0;
        for (let k of Object.keys(this.voc)) {
            if (this.voc[k].id > maxId)
                maxId = this.voc[k].id;
            const { stats } = this.voc[k];
            let values = Object.entries(stats);
            values = values.map((item) => {
                item[1] = item[1] / sizes[Number(item[0])];
                return item;
            });
            values.sort((a, b) => b[1] - a[1]);
            const alpha = values[0];
            const beta = values[1];
            this.voc[k].output = Number(alpha[0]);
            if (beta && beta[1]) {
                const weight = alpha[1] / beta[1];
                if (weight === Infinity)
                    console.warn(weight, alpha, beta);
                this.voc[k].value = weight;
            }
            else {
                this.voc[k].value = -1;
            }
        }
        // add new ids
        for (let k of Object.keys(this.voc)) {
            if (this.voc[k].id === -1) {
                maxId++;
                this.voc[k].id = maxId;
            }
            this.vocValues[this.voc[k].id] = this.voc[k];
        }
        // get median max weight
        let weights = Object.values(this.voc).map((item) => {
            return item.value;
        });
        weights = weights.filter((w) => w !== -1);
        this.maxWeight = this.getMedian(weights, this.median, true);
        for (const key in this.voc) {
            if (this.voc[key].value === -1) {
                this.voc[key].value = this.maxWeight;
            }
            this.voc[key].stats.forEach((val, i) => {
                this.balance[i] = (this.balance[i] || 0) + (val || 0);
            });
        }
        const q = this.balance.reduce((p, a) => p + a, 0) / this.balance.length;
        this.balance = this.balance.map((i) => q / i);
        this.initValue = 1 / this.outputs.length;
        console.log("Training balance", this.balance);
        console.log("LOG:", "Vocabulary is ready. Current size:", size);
        console.timeEnd("voc");
    }
    _train(dataset, iteration = 0) {
        console.time("train");
        this.makeVocabulary(dataset); // edit
        console.log("LOG:", "Training model. Iteration:", iteration);
        const accuracy = [0, 0]; // [exact,total]
        this.notPredicted = [];
        this.predictedValues = [];
        this.predictedBetas = [];
        dataset.forEach((entry) => {
            const tokenized = this.tokenizeMessage(entry.input);
            if (tokenized.length < 2)
                return;
            entry.output = Number(entry.output);
            const result = this.predict(entry.input, true);
            if ("beta" in result) {
                this.predictedValues.push(result.max);
                this.predictedBetas.push(result.beta);
            }
            const predicted = result.output === entry.output;
            accuracy[1]++;
            if (predicted) {
                accuracy[0]++;
            }
            else {
                this.notPredicted.push(entry);
            }
            const layerized = this.layerize(tokenized);
            layerized.forEach((token) => {
                if (!this.model[token]) {
                    this.model[token] = [];
                    this.outputs.forEach((output) => {
                        const dlr = this.dlrCache[token];
                        const modelize = this.getDiff(dlr[0], dlr[1]);
                        if (!this.maxWeight)
                            console.warn("!!!", this.maxWeight);
                        const mr = modelize.result === output ? modelize.weight / this.maxWeight : 0;
                        this.model[token][output] = mr > 1 ? 1 : mr;
                    });
                }
                const value = this.model[token];
                // edit
                Object.keys(value).forEach((k) => {
                    const key = Number(k);
                    const sign = key === entry.output ? 1 : 0;
                    const multiplier = predicted ? 1 : 3;
                    value[key] += sign * multiplier * Math.random();
                    if (value[key] < 0)
                        value[key] = 0;
                    value[key] = Number(value[key].toFixed(3));
                });
            });
        });
        this.thresholds.valueThreshold = this.getMedian(this.predictedValues, this.median, false);
        this.thresholds.betasThreshold = this.getMedian(this.predictedBetas, this.median, false);
        console.timeEnd("train");
        return accuracy[0] / accuracy[1];
    }
    /**
     *
     * @param dataset
     */
    train(dataset) {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                let iteration = 0;
                let acc = 0;
                let cond1, cond2, cond3, cond4;
                do {
                    const acc = this._train(dataset, iteration);
                    if (this.modelAccuracy === acc) {
                        this.accuracyRepeats++;
                    }
                    else {
                        this.accuracyRepeats = 0;
                    }
                    console.log("LOG:", `Training accuracy: ${acc}. Duplication: ${this.accuracyRepeats}.`);
                    cond1 = acc < this.trainingThreshold;
                    cond2 = this.accuracyRepeats < this.accuracyRepeatsStopThreshold;
                    cond3 = acc !== 1;
                    cond4 = this.modelAccuracy < acc;
                    this.modelAccuracy = acc;
                    iteration++;
                } while (cond1 && cond2 && cond3 && cond4);
                this.ready = true;
                const result = {
                    accuracy: acc,
                    iterations: iteration,
                    notPredicted: this.notPredicted,
                };
                return result;
            }
            catch (err) {
                return Promise.reject(err);
            }
        });
    }
    predict(message, auto = false) {
        const response = {
            output: -1,
            max: 0,
            result: [],
        };
        const model = this.model;
        if (!model[Object.keys(model)[0]])
            return response;
        const tokenized = this.tokenizeMessage(message);
        if (tokenized.length < 2)
            return response;
        const layerized = this.layerize(tokenized);
        const result = [];
        for (let i = 0; i < this.outputs.length; i++) {
            let q = 0;
            layerized.forEach((token) => {
                const values = model[token];
                let addition = 0;
                if (values) {
                    const sum = Object.values(values).reduce((p, a) => p + a, 0);
                    if (!sum)
                        console.warn("sum", sum, values);
                    addition = values[i] / sum;
                }
                else {
                    addition = 0;
                }
                q += addition * this.balance[i];
            });
            result[i] = q; // / total || 0
        }
        const orig = [...result];
        result.sort((a, b) => b - a);
        const max = result[0];
        const final = {
            max,
            output: max ? orig.indexOf(max) : -1,
            result: orig,
            beta: result[0] / result[1],
            delta: result[0] / result[result.length - 1],
        };
        if (!auto)
            final.thresholds = this.thresholds;
        return final;
    }
    loadModel(path) {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                const file = yield promises_1.default.readFile(path);
                const parsed = JSON.parse(file.toString());
                this.voc = parsed.voc;
                Object.values(this.voc).forEach((item) => {
                    this.vocValues[item.id] = item;
                });
                this.model = parsed.model;
                this.outputs = parsed.outputs || [0, 1]; // edit
                this.thresholds = parsed.thresholds || {};
                this.initValue = 1 / this.outputs.length;
                this.ready = true;
                console.log("LOG:", "Model successfully loaded!", `Voc size: ${Object.keys(this.voc).length}`, `Model size: ${Object.keys(this.model).length}`, `Outputs: ${this.outputs.length}`);
                return true;
            }
            catch (err) {
                return Promise.reject(err);
            }
        });
    }
    saveModel(path) {
        return __awaiter(this, void 0, void 0, function* () {
            return promises_1.default.writeFile(path, JSON.stringify({
                model: this.model,
                voc: this.voc,
                outputs: this.outputs,
                accuracy: this.modelAccuracy,
                thresholds: this.thresholds,
            }));
        });
    }
}
exports.default = TextClassifier;
