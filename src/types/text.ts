export namespace TextClassifierType {
  export interface InputEntry {
    input: string;
    output: number;
    id?: string;
  }

  export type Dataset = InputEntry[];

  export type Stemmer = (word: string) => string;

  export type Token = string;

  export type Diff = {
    result: number;
    weight: number;
  };

  export interface ClassParams {
    stemmer?: Stemmer;
    trainingThreshold?: number;
    modelizeConstant?: number;
    cleanReg?: RegExp;
    medianMaxWeight?: number;
    medianMinThreshold?: number;
    diffMaxValue?: number;
    predictedWeightMultiplier?: number;
  }

  export type TokenizedMessage = number[];

  export namespace Vocabulary {
    export type StemmedWord = string;
    export type EntryStats = number[];
    export interface Entry {
      id: number;
      output: number;
      value: number;
      stats: EntryStats;
    }
    export interface Storage {
      [key: StemmedWord]: Entry;
    }
  }

  export namespace Model {
    export type OutputWeight = number;
    export interface Storage {
      [key: Token]: OutputWeight[];
    }
    export interface TrainResult {
      accuracy: number;
      iterations: number;
      notPredicted: InputEntry[];
    }
  }

  export namespace Prediction {
    export type Thresholds = {
      valueThreshold: number | null;
      betasThreshold: number | null;
    };

    export interface ResultPositive {
      max: number;
      output: number;
      result: number[];
      beta: number;
      delta: number;
      thresholds?: Thresholds;
    }

    export interface ResultNegative {
      output: -1;
      max: 0;
      result: never[];
    }

    export type Result = ResultPositive | ResultNegative;
  }
}
