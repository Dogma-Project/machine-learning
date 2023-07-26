const fs = require("node:fs/promises");

class TextClassifier {
  /**
   *
   * @param {Object} params
   * @param {Function} params.stemmer
   */
  constructor({
    stemmer,
    trainingThreshold,
    modelizeConstant,
    cleanReg,
    median,
  }) {
    this.voc = {};
    this.vocValues = [];
    this.dlrCache = {};
    this.model = {};
    this.outputs = [];
    this.initValue = 0.5;
    this.ready = false;
    this._modelAccuracy = -1;
    this._accuracyRepeats = 0;
    this._accuracyRepeatsStopThreshold = 10;
    this._diffCache = {};
    this._maxWeight = 1;
    this._thresholds = {
      valueThreshold: null,
      betasThreshold: null,
    };
    this.notPredicted = [];
    this.predictedValues = [];
    this.predictedBetas = [];
    // configs
    this.stemmer = stemmer || this._pseudoStemmer;
    this.trainingThreshold = trainingThreshold || 0.99;
    this.modelizeConstant = modelizeConstant || 0.7;
    this.cleanReg = cleanReg || /[^a-z0-9\ ']+/gi;
    this.median = median || 0.05;
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
   * @param {String} msg
   * @returns {Array}
   */
  _tokenizeMessage(msg) {
    const arr = msg
      .replace(this.cleanReg, " ")
      .split(" ")
      .map((word) => {
        if (!word.length) return -1;
        return this.voc[this.stemmer(word)]
          ? this.voc[this.stemmer(word)].id
          : 0;
      });
    return arr.filter((token) => token !== -1);
  }

  /**
   *
   * @param {Array} tokenized
   * @returns {Array}
   */
  _layerize(tokenized) {
    tokenized = tokenized.map((word, i, arr) => {
      if (i < tokenized.length - 1) {
        const val = `${word}:${arr[i + 1]}`;
        this.dlrCache[val] = [word, arr[i + 1]];
        return val;
      }
    });
    return tokenized.filter((j) => j !== undefined);
  }

  _getDiff(a, b) {
    const key = a + ":" + b;
    if (this._diffCache[key]) return this._diffCache[key];
    const valA = this.vocValues[a];
    const valB = this.vocValues[b];

    let result = -1;
    let weight = 0;

    if (valA && valB) {
      if (valA.output === valB.output) {
        result = valA.output;
        weight = valA.value; // valA.value + valB.value
      } else if (valA.value > valB.value) {
        result = valA.output;
        weight = valA.value - valB.value;
      } else {
        result = valB.output;
        weight = valB.value - valA.value;
      }
    } else if (valA) {
      result = valA.output;
      weight = valA.value;
    } else if (valB) {
      result = valB.output;
      weight = valB.value;
    }

    const res = { result, weight };
    this._diffCache[key] = res;
    return res;
  }

  /**
   *
   * @param {Array} values
   * @param {Number} q
   * @param {Boolean} max default:true
   * @returns
   */
  _getMedian(values, q, max = true) {
    if (max) {
      values.sort((a, b) => b - a);
    } else {
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
  _makeVocabulary(dataset) {
    console.time("voc");
    console.log("LOG:", "Making vocabulary...");

    // prepare and count new data
    const temp = {};
    dataset.forEach((row) => {
      try {
        const arr = row.input.replace(this.cleanReg, " ").split(" ");
        if (!arr.length) return;
        arr.forEach((word) => {
          word = this.stemmer(word);
          if (!word.length) return;
          if (!temp[word]) temp[word] = {};
          const val = temp[word][row.output];
          temp[word][row.output] = (val || 0) + 1;
        });
      } catch (err) {
        console.error(err);
      }
    });

    // add new entries to vocabulary
    for (let k of Object.keys(temp)) {
      if (!this.voc[k]) this.voc[k] = { stats: temp[k] };
    }

    // get new size
    const size = Object.keys(this.voc).length;

    // get size by outputs
    const sizes = {};
    Object.values(this.voc).forEach((item) => {
      const { stats } = item;
      for (let k in stats) {
        sizes[k] = (sizes[k] || 0) + Number(stats[k]);
      }
    });

    let maxId = 0;
    for (let k of Object.keys(this.voc)) {
      if (this.voc[k].id > maxId) maxId = this.voc[k].id;
      const stats = this.voc[k].stats;
      let values = Object.entries(stats);
      values = values.map((item) => {
        item[1] = item[1] / sizes[item[0]];
        return item;
      });
      values.sort((a, b) => b[1] - a[1]);
      const alpha = values[0];
      const beta = values[1];
      this.voc[k].output = Number(alpha[0]);
      if (beta) {
        const weight = alpha[1] / beta[1];
        this.voc[k].value = weight;
      } else {
        this.voc[k].value = -1;
      }
    }

    // add new ids
    for (let k of Object.keys(this.voc)) {
      if (this.voc[k].id === undefined) {
        maxId++;
        this.voc[k].id = maxId;
      }
      this.vocValues[this.voc[k].id] = this.voc[k];
    }

    // get median max weight
    let weights = Object.values(this.voc).map((item) => {
      return item.value;
    });

    this._maxWeight = this._getMedian(weights, this.median, true);

    for (let k of Object.keys(this.voc)) {
      if (this.voc[k].value === -1) this.voc[k].value = this._maxWeight;
    }

    // set outputs
    const outputs = new Set();
    dataset.forEach((entry) => outputs.add(entry.output));
    this.outputs.forEach((entry) => outputs.add(entry));
    this.outputs = [...outputs];
    this.initValue = 1 / this.outputs.length;

    console.log("LOG:", "Vocabulary is ready. Current size:", size);
    console.timeEnd("voc");
  }

  _modelize() {}

  _train(dataset, iteration = 0) {
    console.time("train");
    this._makeVocabulary(dataset); // edit
    console.log("LOG:", "Training model. Iteration:", iteration);
    const accuracy = [0, 0]; // [exact,total]
    this.notPredicted = [];
    this.predictedValues = [];
    this.predictedBetas = [];
    dataset.forEach((entry) => {
      const tokenized = this._tokenizeMessage(entry.input);
      if (tokenized.length < 2) return;
      entry.output = Number(entry.output);
      const result = this.predict(entry.input, true);
      this.predictedValues.push(result.max);
      this.predictedBetas.push(result.beta);
      const predicted = result.output === entry.output;
      accuracy[1]++;
      if (predicted) {
        accuracy[0]++;
      } else {
        this.notPredicted.push(entry);
      }
      const layerized = this._layerize(tokenized);
      layerized.forEach((token) => {
        if (!this.model[token]) {
          this.model[token] = {};
          this.outputs.forEach((output) => {
            const dlr = this.dlrCache[token];
            const modelize = this._getDiff(dlr[0], dlr[1]);
            const mr =
              modelize.result === output
                ? modelize.weight / this._maxWeight
                : 0;
            this.model[token][output] = mr > 1 ? 1 : mr;
          });
        }
        const value = this.model[token];
        // edit
        Object.keys(value).forEach((key) => {
          key = Number(key);
          const sign = key === entry.output ? 1 : 0;
          const multiplier = predicted ? 1 : 5;
          value[key] += sign * multiplier * Math.random();
          if (value[key] < 0) value[key] = 0;
          value[key] = Number(value[key].toFixed(3));
        });
      });
    });
    this._thresholds.valueThreshold = this._getMedian(
      this.predictedValues,
      this.median,
      false
    );
    this._thresholds.betasThreshold = this._getMedian(
      this.predictedBetas,
      this.median,
      false
    );
    console.timeEnd("train");
    return accuracy[0] / accuracy[1];
  }

  /**
   *
   * @param {Array} dataset
   */
  train(dataset) {
    return new Promise((resolve, _reject) => {
      let iteration = 0;
      let acc = 0;
      let cond1, cond2, cond3, cond4;
      do {
        const acc = this._train(dataset, iteration);
        if (this._modelAccuracy === acc) {
          this._accuracyRepeats++;
        } else {
          this._accuracyRepeats = 0;
        }
        console.log(
          "LOG:",
          `Training accuracy: ${acc}. Duplication: ${this._accuracyRepeats}.`
        );
        cond1 = acc < this.trainingThreshold;
        cond2 = this._accuracyRepeats < this._accuracyRepeatsStopThreshold;
        cond3 = acc !== 1;
        cond4 = this._modelAccuracy < acc;
        this._modelAccuracy = acc;
        iteration++;
      } while (cond1 && cond2 && cond3 && cond4);
      this.ready = true;
      resolve({
        accuracy: acc,
        iterations: iteration,
        notPredicted: this.notPredicted,
      });
    });
  }

  predict(message, auto = false) {
    const response = {
      output: -1,
      max: 0,
      result: [],
    };
    const model = this.model;
    if (!model[Object.keys(model)[0]]) return response; // check
    const tokenized = this._tokenizeMessage(message);
    if (tokenized.length < 2) return response;
    const layerized = this._layerize(tokenized);
    const result = [];
    for (let i = 0; i < this.outputs.length; i++) {
      let q = 0;
      layerized.forEach((token) => {
        const values = model[token];
        let addition = 0;
        if (values) {
          const sum = Object.values(values).reduce((p, a) => p + a, 0);
          if (!sum) console.log("sum", sum);
          addition = values[i] / sum;
        } else {
          addition = 0;
        }
        q += addition;
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
    if (!auto) final.thresholds = this._thresholds;
    return final;
  }

  async loadModel(path) {
    try {
      const file = await fs.readFile(path);
      const parsed = JSON.parse(file);
      this.voc = parsed.voc;
      Object.values(this.voc).forEach((item) => {
        this.vocValues[item.id] = item;
      });
      this.model = parsed.model;
      this.outputs = parsed.outputs || [0, 1]; // edit
      this._thresholds = parsed.thresholds || {};
      this.initValue = 1 / this.outputs.length;
      this.ready = true;
      console.log(
        "LOG:",
        "Model successfully loaded!",
        `Voc size: ${Object.keys(this.voc).length}`,
        `Model size: ${Object.keys(this.model).length}`,
        `Outputs: ${this.outputs.length}`
      );
    } catch (err) {
      return err;
    }
  }

  async saveModel(path) {
    return fs.writeFile(
      path,
      JSON.stringify({
        model: this.model,
        voc: this.voc,
        outputs: this.outputs,
        accuracy: this._modelAccuracy,
        thresholds: this._thresholds,
      })
    );
  }
}

module.exports = TextClassifier;
