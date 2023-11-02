"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MarkovGen = exports.TextClassifier = void 0;
const text_classifier_1 = __importDefault(require("./text-classifier"));
exports.TextClassifier = text_classifier_1.default;
const markov_1 = __importDefault(require("./markov"));
exports.MarkovGen = markov_1.default;
