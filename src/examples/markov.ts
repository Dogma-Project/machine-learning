import { MarkovGen } from "../index";

const markov = new MarkovGen({
  input: ["array of sentences", "to base the chains on", "should go here"],
  minLength: 10,
});

markov.makeChain().then(console.log).catch(console.error);
