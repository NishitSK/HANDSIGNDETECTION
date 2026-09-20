// Checks mobile/app/grammar.mjs against output from src/grammar_correction.py.
// Regenerate grammar_reference.json from the Python whenever the rules change.
import { readFileSync } from 'node:fs';
import { correctSentence, processSequence } from '../app/grammar.mjs';

const reference = JSON.parse(readFileSync(new URL('./grammar_reference.json', import.meta.url), 'utf8'));

let total = 0;
let failures = 0;
const check = (name, fn, cases) => {
  for (const { input, output } of cases) {
    total++;
    const actual = fn(input);
    if (actual !== output) {
      failures++;
      console.log(`${name}(${JSON.stringify(input)}): expected ${JSON.stringify(output)}, got ${JSON.stringify(actual)}`);
    }
  }
};

check('correctSentence', correctSentence, reference.correct_sentence);
check('processSequence', processSequence, reference.process_sequence);

console.log(`${total - failures}/${total} match the Python reference`);
process.exit(failures ? 1 : 0);
