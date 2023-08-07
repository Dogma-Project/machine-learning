import { TextClassifierType } from "./types/text";
declare class TextClassifier {
    ready: boolean;
    private voc;
    private vocValues;
    private dlrCache;
    private diffCache;
    private model;
    private outputs;
    private modelAccuracy;
    private accuracyRepeats;
    private accuracyRepeatsStopThreshold;
    private maxWeight;
    private thresholds;
    private notPredicted;
    private predictedValues;
    private predictedBetas;
    private stemmer;
    private trainingThreshold;
    private cleanReg;
    private median;
    private initValue;
    private modelizeConstant;
    private balance;
    private vocabulary;
    /**
     *
     * @param params
     * @param params.stemmer word stemming function
     * @param params.trainingThreshold
     * @param params.modelizeConstant
     * @param params.cleanReg regexp to clean text. default: /[^a-z0-9\ ']+/gi
     * @param params.median
     */
    constructor({ stemmer, trainingThreshold, modelizeConstant, cleanReg, median, }: TextClassifierType.ClassParams);
    /**
     *
     * @param input
     */
    private pseudoStemmer;
    /**
     *
     * @param msg
     */
    private tokenizeMessage;
    /**
     *
     * @param tokenized
     */
    private layerize;
    private getDiff;
    /**
     *
     * @param values
     * @param q
     * @param max default:true
     * @returns
     */
    private getMedian;
    /**
     *
     * @param {Array} data {input, output}
     *
     */
    private makeVocabulary;
    /**
     *
     * @param data [{input, output}]
     *
     */
    private handleDataset;
    private _train;
    /**
     *
     * @param dataset
     */
    train(dataset: TextClassifierType.Dataset): Promise<TextClassifierType.Model.TrainResult>;
    predict(message: string, auto?: boolean): TextClassifierType.Prediction.Result;
    loadModel(path: string): Promise<TextClassifier>;
    saveModel(path: string): Promise<void>;
}
export default TextClassifier;
