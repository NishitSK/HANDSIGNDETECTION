"""
Grammar Correction Module
Converts sign language word sequences to grammatically correct sentences
Uses transformer models for natural language generation
"""

import re


class GrammarCorrector:
    """Grammar correction and sentence formation for ISL translations"""
    
    def __init__(self, model_name="google/flan-t5-base"):
        """
        Initialize grammar correction model
        
        Args:
            model_name: HuggingFace model name (T5, BART, etc.)
        """
        print(f"Loading grammar correction model: {model_name}")
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device = "cpu"

        try:
            auto_tokenizer, auto_model, torch_module = self._load_model_dependencies()
            self.tokenizer = auto_tokenizer.from_pretrained(model_name)
            self.model = auto_model.from_pretrained(model_name)
            self.torch = torch_module
            self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"✓ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            print("Falling back to rule-based grammar correction")

    def _load_model_dependencies(self):
        """Import heavy NLP dependencies lazily so app startup can recover gracefully."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        return AutoTokenizer, AutoModelForSeq2SeqLM, torch
    
    def correct_sentence(self, word_sequence, temperature=0.7, max_length=100, use_model=False):
        """
        Convert word sequence to grammatically correct sentence
        
        Args:
            word_sequence: List of words or space-separated string
            temperature: Sampling temperature (higher = more creative)
            max_length: Maximum output length
            use_model: Force using AI model (False = use rule-based for ISL)
        
        Returns:
            Corrected sentence as string
        """
        if isinstance(word_sequence, list):
            word_sequence = ' '.join(word_sequence)
        
        # For ISL, rule-based is more reliable than AI model
        # AI models tend to be too creative and add unwanted words
        if use_model and self.model and self.tokenizer:
            return self._model_based_correction(word_sequence, temperature, max_length)
        else:
            return self._rule_based_correction(word_sequence)
    
    def _model_based_correction(self, text, temperature, max_length):
        """Model-based grammar correction using transformers"""
        # Create prompt for grammar correction
        prompt = f"Convert this Indian Sign Language word sequence into a grammatically correct sentence suitable for professional use: {text}"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                num_beams=4,
                early_stopping=True,
                do_sample=temperature > 0
            )
        
        # Decode
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return corrected.strip()
    
    def _rule_based_correction(self, text):
        """
        Rule-based grammar correction for fallback
        Simple heuristics for ISL to English conversion
        """
        words = text.strip().split()
        
        if not words:
            return ""
        
        # Convert to lowercase for easier matching
        words = [w.lower() for w in words]
        
        # ISL grammar patterns - apply phrase construction rules
        corrected_sentence = self._construct_phrase(words)
        
        # Capitalize first letter
        corrected_sentence = corrected_sentence[0].upper() + corrected_sentence[1:] if corrected_sentence else ""
        
        # Add punctuation based on question words
        if any(word in words for word in ['who', 'what', 'where', 'when', 'why', 'how']):
            if corrected_sentence and corrected_sentence[-1] not in '.!?':
                corrected_sentence += '?'
        else:
            if corrected_sentence and corrected_sentence[-1] not in '.!?':
                corrected_sentence += '.'
        
        # Fix common patterns
        corrected_sentence = self._apply_grammar_rules(corrected_sentence)
        
        return corrected_sentence
    
    def _construct_phrase(self, words):
        """
        Construct grammatically correct phrases from sign language word sequences
        
        Examples:
        - who you → who are you
        - what name → what is your name
        - how you → how are you
        - where you → where are you
        - when come → when will you come
        """
        # Common phrase patterns in ISL
        phrase_patterns = {
            # Question patterns (wh-word + pronoun)
            ('who', 'you'): 'who are you',
            ('what', 'you'): 'what do you want',
            ('what', 'name'): 'what is your name',
            ('what', 'your', 'name'): 'what is your name',
            ('where', 'you'): 'where are you',
            ('when', 'you'): 'when are you coming',
            ('why', 'you'): 'why are you here',
            ('how', 'you'): 'how are you',
            ('how', 'many'): 'how many',
            ('how', 'much'): 'how much',
            
            # Greeting patterns
            ('hello',): 'hello',
            ('thank', 'you'): 'thank you',
            ('sorry',): 'sorry',
            ('please',): 'please',
            ('help',): 'help me',
            ('help', 'me'): 'help me',
            
            # Common statements
            ('i', 'go'): 'I am going',
            ('i', 'come'): 'I am coming',
            ('you', 'come'): 'you are coming',
            ('you', 'go'): 'you are going',
            ('i', 'eat'): 'I am eating',
            ('i', 'drink'): 'I am drinking',
            
            # Yes/No
            ('yes',): 'yes',
            ('no',): 'no',
            ('good',): 'good',
            ('bad',): 'bad',
        }
        
        # Try to match exact patterns first
        words_tuple = tuple(words)
        if words_tuple in phrase_patterns:
            return phrase_patterns[words_tuple]
        
        # Try partial matches (check if beginning matches)
        for pattern, phrase in phrase_patterns.items():
            if len(words) >= len(pattern):
                if words[:len(pattern)] == list(pattern):
                    # Found match, use it and append remaining words
                    remaining = words[len(pattern):]
                    if remaining:
                        return phrase + ' ' + ' '.join(remaining)
                    return phrase
        
        # Apply common grammar rules for unmatched sequences
        result = []
        i = 0
        
        while i < len(words):
            word = words[i]
            
            # Add "are" after question words followed by pronouns
            if word in ['who', 'what', 'where', 'when', 'why', 'how']:
                result.append(word)
                if i + 1 < len(words) and words[i + 1] in ['you', 'we', 'they']:
                    result.append('are')
                elif i + 1 < len(words) and words[i + 1] in ['he', 'she', 'it']:
                    result.append('is')
                elif i + 1 < len(words) and words[i + 1] == 'i':
                    result.append('am')
            
            # Add helping verbs for pronouns
            elif word in ['i', 'you', 'we', 'they', 'he', 'she', 'it']:
                result.append(word)
                # Check if next word is a verb and needs helping verb
                if i + 1 < len(words) and words[i + 1] in ['go', 'come', 'eat', 'drink', 'sleep', 'sit', 'stand', 'walk']:
                    if word == 'i':
                        result.append('am')
                    elif word in ['you', 'we', 'they']:
                        result.append('are')
                    elif word in ['he', 'she', 'it']:
                        result.append('is')
            
            else:
                result.append(word)
            
            i += 1
        
        return ' '.join(result)
    
    def _apply_grammar_rules(self, sentence):
        """Apply common grammar rules"""
        # Fix 'a' vs 'an'
        sentence = re.sub(r'\ba ([aeiou])', r'an \1', sentence, flags=re.IGNORECASE)
        
        # Fix double spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        
        # Capitalize proper nouns and I
        sentence = re.sub(r'\bi\b', 'I', sentence)
        
        # Fix question marks for question words
        if any(sentence.lower().startswith(q) for q in ['what', 'where', 'when', 'why', 'how', 'who']):
            if sentence[-1] == '.':
                sentence = sentence[:-1] + '?'
        
        return sentence
    
    def format_for_news(self, sentence, formal=True):
        """
        Format sentence for news channel broadcast
        
        Args:
            sentence: Input sentence
            formal: Whether to use formal language
        
        Returns:
            Formatted sentence suitable for news
        """
        # Convert to formal language if needed
        if formal:
            # Replace informal contractions
            replacements = {
                "don't": "do not",
                "can't": "cannot",
                "won't": "will not",
                "shouldn't": "should not",
                "couldn't": "could not",
                "wouldn't": "would not",
                "isn't": "is not",
                "aren't": "are not",
                "wasn't": "was not",
                "weren't": "were not",
                "hasn't": "has not",
                "haven't": "have not",
                "hadn't": "had not",
                "doesn't": "does not",
                "didn't": "did not",
            }
            
            for informal, formal_version in replacements.items():
                sentence = re.sub(
                    r'\b' + informal + r'\b',
                    formal_version,
                    sentence,
                    flags=re.IGNORECASE
                )
        
        return sentence
    
    def process_sequence(self, word_list, context_aware=True):
        """
        Process a sequence of detected signs into coherent sentence
        
        Args:
            word_list: List of detected signs in order
            context_aware: Whether to maintain context between signs
        
        Returns:
            Corrected sentence
        """
        if not word_list:
            return ""
        
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in word_list:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        # Join and correct
        word_sequence = ' '.join(unique_words)
        corrected = self.correct_sentence(word_sequence)
        
        # Format for news if needed
        corrected = self.format_for_news(corrected, formal=True)
        
        return corrected
    
    def get_alternative_phrasings(self, sentence, num_alternatives=3):
        """
        Generate alternative phrasings of the same sentence
        
        Args:
            sentence: Input sentence
            num_alternatives: Number of alternatives to generate
        
        Returns:
            List of alternative sentences
        """
        if not self.model:
            return [sentence]  # Only one option with rule-based
        
        alternatives = []
        
        for temp in [0.5, 0.7, 0.9]:
            alt = self.correct_sentence(sentence, temperature=temp)
            if alt and alt not in alternatives:
                alternatives.append(alt)
            
            if len(alternatives) >= num_alternatives:
                break
        
        return alternatives if alternatives else [sentence]


# Example usage and testing
if __name__ == "__main__":
    corrector = GrammarCorrector()
    
    # Test cases
    test_sequences = [
        "NEWS IMPORTANT TODAY WEATHER",
        "I HAPPY SEE YOU",
        "WHAT TIME NOW",
        "PLEASE HELP ME UNDERSTAND",
        "GOVERNMENT NEW POLICY ANNOUNCE",
        "INDIA WIN MATCH TODAY",
        "WEATHER RAIN TOMORROW"
    ]
    
    print("\n" + "="*60)
    print("GRAMMAR CORRECTION EXAMPLES")
    print("="*60 + "\n")
    
    for seq in test_sequences:
        corrected = corrector.correct_sentence(seq)
        formatted = corrector.format_for_news(corrected)
        
        print(f"Original:  {seq}")
        print(f"Corrected: {corrected}")
        print(f"Formatted: {formatted}")
        print("-" * 60)
