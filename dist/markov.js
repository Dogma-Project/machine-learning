"use strict";
/**
 * Typescript implementation of https://github.com/maximumdata/markov-generator/
 */
Object.defineProperty(exports, "__esModule", { value: true });
class Markov {
    constructor(props) {
        this.props = props;
        if (!this.props.input) {
            throw new Error("input was empty!");
        }
        this.terminals = {};
        this.startWords = [];
        this.wordStats = {};
        this.bannedTerminals =
            props.bannedTerminals && props.bannedTerminals.length
                ? props.bannedTerminals.map((word) => word.toLowerCase())
                : [];
        this.maxIterations = props.maxIterations || 100;
        this.props.input.forEach((e, i, a) => {
            let words = e.split(" ");
            let lastWord = words[words.length - 1];
            let firstWord = words[0];
            // if this.terminals contains the last word in this sentence, add to it's counter
            // otherwise, create the property on this.terminals and set it to 1
            if (this.terminals[lastWord]) {
                this.terminals[lastWord]++;
            }
            else {
                this.terminals[lastWord] = 1;
            }
            // this function tests to see if this.startWords already contains the first word or not
            // we can't use Array.prototype.includes, because when comparing each element in this.startWords to the first word, we need to compare them as lowercase
            let checkWordNotInStartWords = (refWord) => {
                this.startWords.forEach((elem, ind, arr) => {
                    if (elem.toLowerCase() === firstWord.toLowerCase()) {
                        return false;
                    }
                });
                return true;
            };
            // if the first word is not a space, and if this.startWords does not already contain the first word, add it
            if (firstWord.length && checkWordNotInStartWords(firstWord)) {
                this.startWords.push(firstWord);
            }
            // loop through each word in current sentence
            words.forEach((el, it, ar) => {
                // if this.wordStats already contains the current word in the sentence as a property, push the next word in the sentence to it's array
                // otherwise, create the property on this.startWords and set it to an array containing the next word in the sentence
                // first check to see if there even IS a next word
                // we store all of the keys in this.wordStats as lowercase to make the function makeChain case insensitive
                if (ar[it + 1]) {
                    if (this.wordStats.hasOwnProperty(el.toLowerCase())) {
                        this.wordStats[el.toLowerCase()].push(ar[it + 1]);
                    }
                    else {
                        this.wordStats[el.toLowerCase()] = [ar[it + 1]];
                    }
                }
            });
        });
        for (let word in this.terminals) {
            if (!this.terminals[word]) {
                // this.terminals[word] === "" ||
                delete this.terminals[word];
            }
        }
        delete this.terminals[""];
        delete this.wordStats[""];
    }
    isBannedTerminal(word) {
        return this.bannedTerminals.includes(word.toLowerCase());
    }
    /**
     * Choose a random element in a given array
     * @param a - An array to randomly choose an element from
     * @return The selected element of the array
     */
    choice(a) {
        return a[Math.floor(a.length * Math.random())];
    }
    _makeChain(minLength) {
        try {
            let word = this.choice(this.startWords);
            let chain = [word];
            let iterations = 0;
            while (this.wordStats.hasOwnProperty(word.toLowerCase()) &&
                iterations <= this.maxIterations) {
                let nextWords = this.wordStats[word.toLowerCase()];
                word = this.choice(nextWords);
                chain.push(word);
                if (chain.length > minLength &&
                    this.terminals.hasOwnProperty(word) &&
                    !this.isBannedTerminal(word)) {
                    break;
                }
                iterations++;
            }
            if (iterations === this.maxIterations)
                throw "Not enough training data";
            return chain;
        }
        catch (err) {
            throw err;
        }
    }
    /**
     * Creates a new string via a Markov chain based on the input array from the constructor
     * @param minLength - The minimum number of words in the generated string
     */
    makeChain(minLength = this.props.minLength || 10) {
        try {
            let chain = [];
            let iterations = 0;
            let cond1 = false;
            let cond2 = false;
            let cond3 = false;
            do {
                chain = this._makeChain(minLength);
                iterations++;
                cond1 = iterations <= this.maxIterations;
                cond2 = this.props.input.includes(chain.join(" "));
                cond3 = chain.length < minLength;
            } while (cond1 && (cond2 || cond3));
            if (!cond1)
                return Promise.reject("Not enough training data");
            return Promise.resolve(chain.join(" "));
        }
        catch (err) {
            return Promise.reject(err);
        }
    }
}
exports.default = Markov;
