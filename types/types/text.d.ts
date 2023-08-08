export declare namespace TextClassifierType {
    interface InputEntry {
        input: string;
        output: number;
        id?: string;
    }
    type Dataset = InputEntry[];
    type Stemmer = (word: string) => string;
    type Token = string;
    type Diff = {
        result: number;
        weight: number;
    };
    interface ClassParams {
        stemmer?: Stemmer;
        trainingThreshold?: number;
        modelizeConstant?: number;
        cleanReg?: RegExp;
        medianMaxWeight?: number;
        medianMinThreshold?: number;
    }
    type TokenizedMessage = number[];
    namespace Vocabulary {
        type StemmedWord = string;
        type EntryStats = number[];
        interface Entry {
            id: number;
            output: number;
            value: number;
            stats: EntryStats;
        }
        interface Storage {
            [key: StemmedWord]: Entry;
        }
    }
    namespace Model {
        type OutputWeight = number;
        interface Storage {
            [key: Token]: OutputWeight[];
        }
        interface TrainResult {
            accuracy: number;
            iterations: number;
            notPredicted: InputEntry[];
        }
    }
    namespace Prediction {
        type Thresholds = {
            valueThreshold: number | null;
            betasThreshold: number | null;
        };
        interface ResultPositive {
            max: number;
            output: number;
            result: number[];
            beta: number;
            delta: number;
            thresholds?: Thresholds;
        }
        interface ResultNegative {
            output: -1;
            max: 0;
            result: never[];
        }
        type Result = ResultPositive | ResultNegative;
    }
}
