// Port of the rule-based path in src/grammar_correction.py. That module also loads a
// 948 MB Flan-T5 model, but correct_sentence() defaults to use_model=False, so the
// rules below are the behaviour the desktop app actually ships.
// tests/verify_grammar.mjs checks this against the Python output.

const QUESTION_WORDS = ['who', 'what', 'where', 'when', 'why', 'how'];
const PRONOUNS = ['i', 'you', 'we', 'they', 'he', 'she', 'it'];
const ACTION_VERBS = ['go', 'come', 'eat', 'drink', 'sleep', 'sit', 'stand', 'walk'];

// Order matters: prefix matching takes the first pattern that fits, as the Python dict does.
const PHRASE_PATTERNS = [
  [['who', 'you'], 'who are you'],
  [['what', 'you'], 'what do you want'],
  [['what', 'name'], 'what is your name'],
  [['what', 'your', 'name'], 'what is your name'],
  [['where', 'you'], 'where are you'],
  [['when', 'you'], 'when are you coming'],
  [['why', 'you'], 'why are you here'],
  [['how', 'you'], 'how are you'],
  [['how', 'many'], 'how many'],
  [['how', 'much'], 'how much'],
  [['hello'], 'hello'],
  [['thank', 'you'], 'thank you'],
  [['sorry'], 'sorry'],
  [['please'], 'please'],
  [['help'], 'help me'],
  [['help', 'me'], 'help me'],
  [['i', 'go'], 'I am going'],
  [['i', 'come'], 'I am coming'],
  [['you', 'come'], 'you are coming'],
  [['you', 'go'], 'you are going'],
  [['i', 'eat'], 'I am eating'],
  [['i', 'drink'], 'I am drinking'],
  [['yes'], 'yes'],
  [['no'], 'no'],
  [['good'], 'good'],
  [['bad'], 'bad'],
];

const FORMAL_REPLACEMENTS = [
  ["don't", 'do not'],
  ["can't", 'cannot'],
  ["won't", 'will not'],
  ["shouldn't", 'should not'],
  ["couldn't", 'could not'],
  ["wouldn't", 'would not'],
  ["isn't", 'is not'],
  ["aren't", 'are not'],
  ["wasn't", 'was not'],
  ["weren't", 'were not'],
  ["hasn't", 'has not'],
  ["haven't", 'have not'],
  ["hadn't", 'had not'],
  ["doesn't", 'does not'],
  ["didn't", 'did not'],
];

const startsWith = (words, pattern) =>
  pattern.length <= words.length && pattern.every((word, i) => words[i] === word);

function constructPhrase(words) {
  const exact = PHRASE_PATTERNS.find(([pattern]) => pattern.length === words.length && startsWith(words, pattern));
  if (exact) return exact[1];

  for (const [pattern, phrase] of PHRASE_PATTERNS) {
    if (startsWith(words, pattern)) {
      const remaining = words.slice(pattern.length);
      return remaining.length ? `${phrase} ${remaining.join(' ')}` : phrase;
    }
  }

  const result = [];
  words.forEach((word, i) => {
    const next = words[i + 1];
    result.push(word);
    if (QUESTION_WORDS.includes(word)) {
      if (['you', 'we', 'they'].includes(next)) result.push('are');
      else if (['he', 'she', 'it'].includes(next)) result.push('is');
      else if (next === 'i') result.push('am');
    } else if (PRONOUNS.includes(word) && ACTION_VERBS.includes(next)) {
      if (word === 'i') result.push('am');
      else if (['you', 'we', 'they'].includes(word)) result.push('are');
      else result.push('is');
    }
  });
  return result.join(' ');
}

function applyGrammarRules(sentence) {
  let s = sentence.replace(/\ba ([aeiou])/gi, 'an $1');
  s = s.replace(/\s+/g, ' ');
  s = s.replace(/\bi\b/g, 'I');
  if (QUESTION_WORDS.some((q) => s.toLowerCase().startsWith(q)) && s.endsWith('.')) {
    s = `${s.slice(0, -1)}?`;
  }
  return s;
}

export function correctSentence(text) {
  const words = text.trim().split(/\s+/).filter(Boolean).map((w) => w.toLowerCase());
  if (!words.length) return '';

  let s = constructPhrase(words);
  s = s ? s[0].toUpperCase() + s.slice(1) : '';
  if (s && !'.!?'.includes(s.at(-1))) {
    s += words.some((w) => QUESTION_WORDS.includes(w)) ? '?' : '.';
  }
  return applyGrammarRules(s);
}

export function formatForNews(sentence) {
  let s = sentence;
  for (const [informal, formal] of FORMAL_REPLACEMENTS) {
    s = s.replace(new RegExp(`\\b${informal}\\b`, 'gi'), formal);
  }
  return s;
}

export function processSequence(signs) {
  if (!signs.length) return '';
  const unique = [...new Set(signs)];
  return formatForNews(correctSentence(unique.join(' ')));
}
