/**
 * Typescript implementation of https://github.com/maximumdata/markov-generator/
 */
type MarkovProperties = {
    input: string[];
    minLength?: number;
    bannedTerminals?: string[];
    maxIterations?: number;
};
declare class Markov {
    /**
     * Builds the generator
     * @param props - The configuration options and input data. Min length = 10
     */
    props: MarkovProperties;
    bannedTerminals: string[];
    startWords: string[];
    wordStats: {
        [index: string]: any;
    };
    terminals: {
        [index: string]: number;
    };
    maxIterations: number;
    constructor(props: MarkovProperties);
    isBannedTerminal(word: string): boolean;
    /**
     * Choose a random element in a given array
     * @param a - An array to randomly choose an element from
     * @return The selected element of the array
     */
    choice(a: string[]): string;
    private _makeChain;
    /**
     * Creates a new string via a Markov chain based on the input array from the constructor
     * @param minLength - The minimum number of words in the generated string
     */
    makeChain(minLength?: number): Promise<string>;
}
export default Markov;
