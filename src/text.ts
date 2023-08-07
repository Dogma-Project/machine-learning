import fs from "node:fs/promises";
import { TextClassifierType } from "./types/text";

class TextClassifier {
  ready: boolean;

  private voc: TextClassifierType.Vocabulary.Storage;
  private vocValues: TextClassifierType.Vocabulary.Entry[];
  private dlrCache: {
    [key: TextClassifierType.Token]: [number, number];
  };
  private diffCache: {
    [key: TextClassifierType.Token]: TextClassifierType.Diff;
  };
  private model: TextClassifierType.Model.Storage;
  private outputs: number[];
  private modelAccuracy: number;
  private accuracyRepeats: number;
  private accuracyRepeatsStopThreshold: number;
  private maxWeight: number;
  private thresholds: TextClassifierType.Prediction.Thresholds;
  private notPredicted: TextClassifierType.InputEntry[];
  private predictedValues: number[];
  private predictedBetas: number[];
  private stemmer: TextClassifierType.Stemmer;
  private trainingThreshold: number;
  private cleanReg: RegExp;
  private median: number;
  private initValue: number;
  private modelizeConstant: number;
  private balance: number[];
  private vocabulary: string[];
  private modelVersion: number;

  /**
   *
   * @param params
   * @param params.stemmer word stemming function
   * @param params.trainingThreshold
   * @param params.modelizeConstant
   * @param params.cleanReg regexp to clean text. default: /[^a-z0-9\ ']+/gi
   * @param params.median
   */
  constructor({
    stemmer,
    trainingThreshold,
    modelizeConstant,
    cleanReg,
    median,
  }: TextClassifierType.ClassParams) {
    // configs
    this.stemmer = stemmer || this.pseudoStemmer;
    this.trainingThreshold = trainingThreshold || 0.99;
    this.modelizeConstant = modelizeConstant || 0.7;
    this.cleanReg = cleanReg || /[^a-z0-9\ ']+/gi;
    this.median = median || 0.05;

    this.vocabulary = [];
    // add cache
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
    this.modelVersion = 10;
  }

  /**
   *
   * @param input
   */
  private pseudoStemmer(input: string) {
    return input.toLowerCase();
  }

  /**
   *
   * @param msg
   */
  private tokenizeMessage(msg: string) {
    const arr = msg
      .replace(this.cleanReg, " ")
      .split(" ")
      .map((word) => {
        if (!word.length) return -1;
        word = this.stemmer(word);
        return this.vocabulary.indexOf(word);
      });
    return arr.filter((token) => token !== -1);
  }

  /**
   *
   * @param tokenized
   */
  private layerize(tokenized: TextClassifierType.TokenizedMessage): string[] {
    const result = tokenized.map((word, i, arr) => {
      if (i < tokenized.length - 1) {
        const val = `${word}:${arr[i + 1]}`;
        this.dlrCache[val] = [word, arr[i + 1]];
        return val;
      }
    });
    return result.filter((j) => j !== undefined) as string[];
  }

  private getDiff(a: number, b: number) {
    const key = a + ":" + b;
    if (this.diffCache[key]) return this.diffCache[key];
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
  private getMedian(values: number[], q: number, max: boolean = true) {
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
  private makeVocabulary(dataset: TextClassifierType.Dataset) {
    console.time("voc");
    console.log("LOG:", "Making vocabulary...");
    dataset.forEach((row) => {
      try {
        const arr = row.input.replace(this.cleanReg, " ").split(" ");
        if (!arr.length) return;
        arr.forEach((word) => {
          word = this.stemmer(word);
          if (!word.length) return;
          if (this.vocabulary.indexOf(word) === -1) {
            this.vocabulary.push(word);
          }
        });
      } catch (err) {
        console.error(err);
      }
    });
    console.log(
      "LOG:",
      "Vocabulary is ready. Current size:",
      this.vocabulary.length
    );
    console.timeEnd("voc");
  }

  /**
   *
   * @param data [{input, output}]
   *
   */
  private handleDataset(dataset: TextClassifierType.Dataset) {
    console.time("handle");
    console.log("LOG:", "Handle dataset...");
    this.voc = {};

    // set outputs
    const outputs: Set<number> = new Set();
    dataset.forEach((entry) => outputs.add(entry.output));
    this.outputs.forEach((entry) => outputs.add(entry));
    this.outputs = [...outputs];
    this.initValue = 1 / this.outputs.length;

    // prepare and count new data
    const sizes: number[] = [];
    dataset.forEach((row) => {
      try {
        const arr = row.input.replace(this.cleanReg, " ").split(" ");
        if (!arr.length) return;
        arr.forEach((word) => {
          word = this.stemmer(word);
          if (!word.length) return;
          if (!this.voc[word]) {
            this.voc[word] = {
              id: this.vocabulary.indexOf(word), // edit
              output: 0,
              value: 0,
              stats: Array(this.outputs.length).fill(0),
            };
          }
          const val = this.voc[word].stats[row.output];
          this.voc[word].stats[row.output] = (val || 0) + 1;
          sizes[row.output] = (sizes[row.output] || 0) + 1;
        });
      } catch (err) {
        console.error(err);
      }
    });

    for (const key in this.voc) {
      const { stats } = this.voc[key];
      const values = Object.entries(stats).map((item) => {
        item[1] = item[1] / sizes[Number(item[0])];
        return item;
      });
      values.sort((a, b) => b[1] - a[1]);
      const alpha = values[0];
      const beta = values[1];
      this.voc[key].output = Number(alpha[0]);
      if (beta && beta[1]) {
        const weight = alpha[1] / beta[1];
        if (weight === Infinity) console.warn(weight, alpha, beta);
        this.voc[key].value = Number(weight.toFixed(3));
      } else {
        this.voc[key].value = -1;
      }
    }

    // get median max weight
    let weights = Object.values(this.voc).map((item) => item.value);
    weights = weights.filter((w) => w !== -1);
    this.maxWeight = Number(
      this.getMedian(weights, this.median, true).toFixed(3)
    );

    // set max weight and balances
    for (const key in this.voc) {
      const { id } = this.voc[key];
      if (this.voc[key].value === -1) {
        this.voc[key].value = this.maxWeight;
      }
      this.voc[key].stats.forEach((val, i) => {
        this.balance[i] = (this.balance[i] || 0) + (val || 0);
      });
      this.vocValues[id] = this.voc[key];
    }

    // transform balance
    const q = this.balance.reduce((p, a) => p + a, 0) / this.balance.length;
    this.balance = this.balance.map((i) => q / i);

    console.log("Training balance", this.balance);
    console.log("LOG:", "Dataset is handled.");
    console.timeEnd("handle");
  }

  private _train(dataset: TextClassifierType.Dataset, iteration = 0) {
    console.time("train");
    if (!iteration) this.makeVocabulary(dataset);
    this.handleDataset(dataset);
    console.log("LOG:", "Training model. Iteration:", iteration);
    const accuracy = [0, 0]; // [exact,total]
    this.notPredicted = [];
    this.predictedValues = [];
    this.predictedBetas = [];
    dataset.forEach((entry) => {
      const tokenized = this.tokenizeMessage(entry.input);
      if (tokenized.length < 2) return;
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
      } else {
        this.notPredicted.push(entry);
      }
      const layerized = this.layerize(tokenized);
      layerized.forEach((token) => {
        if (!this.model[token]) {
          this.model[token] = [];
          this.outputs.forEach((output) => {
            const dlr = this.dlrCache[token];
            const modelize = this.getDiff(dlr[0], dlr[1]);
            const mr =
              modelize.result === output ? modelize.weight / this.maxWeight : 0;
            if (!this.maxWeight) console.warn("!!!", this.maxWeight);
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
          if (value[key] < 0) value[key] = 0;
          value[key] = Number(value[key].toFixed(3));
        });
      });
    });
    this.thresholds.valueThreshold = this.getMedian(
      this.predictedValues,
      this.median,
      false
    );
    this.thresholds.betasThreshold = this.getMedian(
      this.predictedBetas,
      this.median,
      false
    );
    console.timeEnd("train");
    return accuracy[0] / accuracy[1];
  }

  /**
   *
   * @param dataset
   */
  async train(dataset: TextClassifierType.Dataset) {
    try {
      let iteration = 0;
      let acc = 0;
      let cond1, cond2, cond3, cond4;
      do {
        const acc = this._train(dataset, iteration);
        if (this.modelAccuracy === acc) {
          this.accuracyRepeats++;
        } else {
          this.accuracyRepeats = 0;
        }
        console.log(
          "LOG:",
          `Training accuracy: ${acc}. Duplication: ${this.accuracyRepeats}.`
        );
        cond1 = acc < this.trainingThreshold;
        cond2 = this.accuracyRepeats < this.accuracyRepeatsStopThreshold;
        cond3 = acc !== 1;
        cond4 = this.modelAccuracy < acc;
        this.modelAccuracy = acc;
        iteration++;
      } while (cond1 && cond2 && cond3 && cond4);
      this.ready = true;
      const result: TextClassifierType.Model.TrainResult = {
        accuracy: acc,
        iterations: iteration,
        notPredicted: this.notPredicted,
      };
      return result;
    } catch (err) {
      return Promise.reject(err);
    }
  }

  predict(message: string, auto = false): TextClassifierType.Prediction.Result {
    const response: TextClassifierType.Prediction.ResultNegative = {
      output: -1,
      max: 0,
      result: [],
    };
    const model = this.model;
    if (!model[Object.keys(model)[0]]) return response;
    const tokenized = this.tokenizeMessage(message);
    if (tokenized.length < 2) return response;
    const layerized = this.layerize(tokenized);
    const result = [];
    for (let i = 0; i < this.outputs.length; i++) {
      let q = 0;
      layerized.forEach((token) => {
        const values = model[token];
        let addition = 0;
        if (values) {
          const sum = Object.values(values).reduce((p, a) => p + a, 0);
          if (sum) {
            addition = values[i] / sum;
          } else {
            console.warn("sum", sum, values);
          }
        }
        q += addition * this.balance[i];
      });
      result[i] = q; // / total || 0
    }
    const orig = [...result];
    result.sort((a, b) => b - a);
    const max = result[0];
    const final: TextClassifierType.Prediction.ResultPositive = {
      max,
      output: max ? orig.indexOf(max) : -1,
      result: orig,
      beta: result[0] / result[1],
      delta: result[0] / result[result.length - 1],
    };
    if (!auto && final.output !== -1) {
      final.thresholds = {
        valueThreshold: null,
        betasThreshold: null,
      };
      const q = this.balance[final.output];
      if (this.thresholds.valueThreshold) {
        final.thresholds.valueThreshold = this.thresholds.valueThreshold / q;
      }
      if (this.thresholds.betasThreshold) {
        final.thresholds.betasThreshold = this.thresholds.betasThreshold / q;
      }
    }
    return final;
  }

  async loadModel(path: string): Promise<TextClassifier> {
    try {
      const file = await fs.readFile(path);
      const parsed = JSON.parse(file.toString());
      this.vocabulary = parsed.vocabulary || [];
      this.model = parsed.model || {};
      this.outputs = parsed.outputs || [0, 1]; // edit
      this.thresholds = parsed.thresholds || {};
      this.balance = parsed.balance || parsed.outputs.map(() => 1);
      this.initValue = 1 / this.outputs.length;
      this.ready = true;
      console.log(
        "LOG:",
        "Model successfully loaded!",
        `Voc size: ${Object.keys(this.vocabulary).length}`,
        `Model size: ${Object.keys(this.model).length}`,
        `Outputs: ${this.outputs.length}`
      );
      return this;
    } catch (err) {
      return Promise.reject(err);
    }
  }

  async saveModel(path: string) {
    return fs.writeFile(
      path,
      JSON.stringify({
        model: this.model,
        vocabulary: this.vocabulary,
        outputs: this.outputs,
        accuracy: this.modelAccuracy,
        thresholds: this.thresholds,
        balance: this.balance,
        modelVersion: this.modelVersion,
      })
    );
  }
}

export default TextClassifier;
