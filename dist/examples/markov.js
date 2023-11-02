"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const index_1 = require("../index");
const markov = new index_1.MarkovGen({
    input: ["array of sentences", "to base the chains on", "should go here"],
    minLength: 10,
});
markov.makeChain().then(console.log).catch(console.error);
