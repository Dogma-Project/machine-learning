const fs = require("node:fs/promises");

class TextClassifier {
  /**
   *
   * @param {Object} params
   * @param {Function} params.stemmer
   */
  constructor({
    stemmer,
    learningRate,
    trainingThreshold,
    minProbability,
    modelizeConstant,
    splitReg,
    cleanReg,
  }) {
    this.vocabulary = {};
    this.plainVocabulary = [];
    this.dlrCache = [];
    this.splitReg = splitReg || /[,.!?]| |\n/;
    this.cleanReg = cleanReg || /[^a-z0-9\ ']+/gi;
    this.model = {};
    this.outputs = [];
    this.initValue = 0.5;
    this.modelAccuracy = 0;
    this.ready = false;
    this._lastAccuracy = -1;
    this._accuracyRepeats = 0;
    this._accuracyRepeatsStopThreshold = 10;
    this._learningAccuracy = 2;
    this._learningAccuracyStep = 0.05;
    // configs
    this.stemmer = stemmer || this._pseudoStemmer;
    this.learningRate = learningRate || 0.03;
    this.trainingThreshold = trainingThreshold || 0.99;
    this.minProbability = minProbability || 0.01;
    this.modelizeConstant = modelizeConstant || 0.7;
    // other
    String.prototype.prepare = function () {
      return this.replace(this.cleanReg, "");
    };
  }

  /**
   *
   * @param {String} msg
   * @returns {Array}
   */
  _tokenizeMessage(msg) {
    const arr = msg.split(this.splitReg).map((word) => {
      const prepared = word.prepare();
      const index = this.plainVocabulary.indexOf(this.stemmer(prepared));
      return index;
    });
    return arr.filter((token) => token !== -1);
  }

  /**
   *
   * @param {String} input
   * @returns {String}
   */
  _pseudoStemmer(input) {
    return input.toLowerCase();
  }

  /**
   *
   * @param {Array} tokenized
   * @returns {Array}
   */
  _layerize(tokenized) {
    tokenized = tokenized.map((word, i, arr) => {
      if (i < tokenized.length - 1) {
        const val = Number(word.toString() + arr[i + 1].toString());
        this.dlrCache[val] = [word, arr[i + 1]];
        return val;
      }
    });
    return tokenized.filter((j) => j !== undefined); // check
  }

  _getValue(token) {
    const val = this.vocabulary.find(
      (item) => item[0] === this.plainVocabulary[token]
    );
    return val ? val[1] : 0; // edit
  }

  _getDiff(a, b) {
    const valA = this._getValue(a);
    const valB = this._getValue(b);
    return (valA + valB) / 2;
  }

  /**
   *
   * @param {Array} data {input, output}
   *
   */
  _makeVocabulary(dataset) {
    console.time("voc");
    console.log("LOG:", "Making vocabulary...");
    const result = {};
    let voc = [];
    dataset.forEach((row) => {
      try {
        const arr = row.input.split(this.splitReg);
        if (!arr.length) return;
        arr.forEach((word) => {
          word = this.stemmer(word.prepare());
          if (!word.length) return;
          if (!result[word]) result[word] = {};
          const obj = result[word] || {};
          obj[row.output] = (obj[row.output] || 0) + 1;
        });
      } catch (err) {
        console.error(err);
      }
    });
    const size = Object.keys(result).length;
    for (let k of Object.keys(result)) {
      const stats = result[k];
      // const sum = Object.values(stats).reduce((p, a) => p + a, 0);
      const max = [-1, 0];
      for (let i of Object.keys(stats)) {
        // edit
        if (stats[i] > max[1]) {
          max[0] = i;
          max[1] = stats[i];
        }
      }
      const entry = [k, max[1] / size + Number(max[0])];
      voc.push(entry);
    }
    this.vocabulary = voc.sort((a, b) => a[1] - b[1]);
    this.plainVocabulary = voc.map((row) => row[0]);

    // set outputs
    const outputs = new Set();
    dataset.forEach((entry) => outputs.add(entry.output));
    this.outputs = [...outputs];
    this.initValue = 1 / this.outputs.length;
    console.log(
      "LOG:",
      "Vocabulary is ready. Current size:",
      this.plainVocabulary.length
    );
    console.timeEnd("voc");
  }

  _modelize() {}

  _train(dataset, iteration = 0) {
    console.time("train");
    this._makeVocabulary(dataset); // ok
    console.log("LOG:", "Training model. Iteration:", iteration);
    const accuracy = [0, 0, 0]; // [cond,exact,total]
    dataset.forEach((entry) => {
      const tokenized = this._tokenizeMessage(entry.input); // ok
      if (tokenized.length < 2) return;
      entry.output = Number(entry.output); // ok
      const result = this.predict(entry.input); // ok
      const predicted = result.output === entry.output;
      if (predicted) accuracy[1]++;
      let ok = false;
      accuracy[2]++;
      if (predicted && result.delta > this._learningAccuracy) {
        ok = true;
        accuracy[0]++;
      }
      const layerized = this._layerize(tokenized); // ok
      layerized.forEach((token) => {
        if (!this.model[token]) {
          this.model[token] = {};
          this.outputs.forEach((output) => {
            this.model[token][output] = this.initValue;
          });
        }
        const value = this.model[token];
        Object.keys(value).forEach((key) => {
          key = Number(key);
          const sign = key !== entry.output ? -1 : 1;
          if (value[key] > 1 - this.minProbability)
            return (value[key] = 1 - this.minProbability);
          if (value[key] < this.minProbability)
            return (value[key] = this.minProbability);
          value[key] =
            value[key] + sign * value[key] * this.learningRate * Math.random();
        });
      });
    });
    console.timeEnd("train");
    return accuracy;
  }

  /**
   *
   * @param {Object} dataset
   */
  train(dataset) {
    return new Promise((resolve, reject) => {
      let iteration = 0;
      let acc = 0;
      let exact = 0;
      let cond1, cond2, cond3;
      do {
        const result = this._train(dataset, iteration);
        acc = result[0] / result[2];
        exact = result[1] / result[2];
        if (acc && this._lastAccuracy === acc) {
          this._accuracyRepeats++;
        } else {
          this._accuracyRepeats = 0;
        }
        console.log("LOG:", "Training accuracy:", acc);
        console.log("LOG:", "Exact training accuracy:", exact);
        console.log("LOG:", "Accuracy repeats", this._accuracyRepeats);
        console.log("LOG:", "Learning accuracy", this._learningAccuracy);
        if (this._learningAccuracy > 1 + this._learningAccuracyStep) {
          // if (this._lastAccuracy < acc) {
          //   this._learningAccuracy += this._learningAccuracyStep;
          // } else if (this._lastAccuracy > acc) {
          //   this._learningAccuracy -= this._learningAccuracyStep;
          // }
          if (this._lastAccuracy >= acc) {
            this._learningAccuracy -= this._learningAccuracyStep;
          }
        }
        this._lastAccuracy = acc;
        this.modelAccuracy = acc;
        iteration++;

        cond1 = acc < this.trainingThreshold;
        cond2 = this._accuracyRepeats < this._accuracyRepeatsStopThreshold;
        cond3 = acc !== 1;
      } while (cond1 && cond2 && cond3);
      this.ready = true;
      resolve({
        accuracy: acc,
        iterations: iteration,
      });
    });
  }

  predict(message) {
    const response = {
      output: -1,
      max: 0,
      result: [],
    };
    const model = this.model;
    if (!model[Object.keys(model)[0]]) return response;
    const tokenized = this._tokenizeMessage(message);
    const layerized = this._layerize(tokenized);
    const result = [];
    for (let i = 0; i < this.outputs.length; i++) {
      let q = 0,
        total = 0;
      layerized.forEach((token) => {
        const values = model[token];
        total++;
        if (values) {
          q += values[i];
        } else {
          const dlr = this.dlrCache[token];
          const modelize = this._getDiff(dlr[0], dlr[1]);
          q += Math.floor(modelize) === i ? this.modelizeConstant : 0;
        }
      });
      result[i] = q / total;
    }
    const max = Math.max(...result);
    return {
      max,
      output: result.indexOf(max),
      result,
      delta: Math.max(...result) / Math.min(...result),
    };
  }

  async loadModel(path) {
    // edit
    const file = await fs.readFile(path);
    const parsed = JSON.parse(file);
    this.vocabulary = parsed.vocabulary;
    this.model = parsed.model;
    this.plainVocabulary = this.vocabulary.map((row) => row[0]);
    this.outputs = parsed.outputs || [0, 1]; // edit
    this.initValue = 1 / this.outputs.length;
    this.ready = true;
    console.log("LOG:", "Model successfully loaded!");
  }

  async saveModel(path) {
    return fs.writeFile(
      path,
      JSON.stringify({
        model: this.model,
        vocabulary: this.vocabulary,
        outputs: this.outputs,
        accuracy: this.modelAccuracy,
      })
    );
  }
}

module.exports = TextClassifier;
