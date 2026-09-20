"""
Test Grammar Correction - Phrase Construction
Shows how the system converts sign sequences to proper sentences
"""

from src.grammar_correction import GrammarCorrector

def test_phrase_construction():
    """Test various sign language sequences"""
    
    print("\n" + "="*70)
    print("ISL GRAMMAR CORRECTION - PHRASE CONSTRUCTION TEST")
    print("="*70 + "\n")
    
    corrector = GrammarCorrector()
    
    # Test cases - Sign sequences that need grammar correction
    test_cases = [
        # Question patterns
        ("who you", "Who are you?"),
        ("what you", "What do you want?"),
        ("what name", "What is your name?"),
        ("where you", "Where are you?"),
        ("when you", "When are you coming?"),
        ("why you", "Why are you here?"),
        ("how you", "How are you?"),
        ("how many", "How many?"),
        ("how much", "How much?"),
        
        # Greetings
        ("hello", "Hello."),
        ("thank you", "Thank you."),
        ("sorry", "Sorry."),
        ("please", "Please."),
        ("help", "Help me."),
        
        # Statements
        ("i go", "I am going."),
        ("you come", "You are coming."),
        ("i eat", "I am eating."),
        
        # Yes/No
        ("yes", "Yes."),
        ("no", "No."),
        ("good", "Good."),
        ("bad", "Bad."),
        
        # Complex sequences
        ("who you where go", "Who are you where go?"),
        ("what you eat", "What do you want eat?"),
        ("i go home", "I am going home."),
    ]
    
    print("Testing phrase construction patterns:\n")
    
    success_count = 0
    total_count = len(test_cases)
    
    for sign_sequence, expected in test_cases:
        result = corrector.correct_sentence(sign_sequence)
        
        # Check if result matches expected (case-insensitive)
        matches = result.lower().strip() == expected.lower().strip()
        status = "✓" if matches else "✗"
        
        if matches:
            success_count += 1
        
        print(f"{status} Input:    '{sign_sequence}'")
        print(f"  Output:   '{result}'")
        if not matches:
            print(f"  Expected: '{expected}'")
        print()
    
    print("-" * 70)
    print(f"\nResults: {success_count}/{total_count} patterns matched correctly")
    print(f"Success Rate: {success_count/total_count*100:.1f}%\n")
    
    # Additional examples showing the system in action
    print("="*70)
    print("REAL-WORLD USAGE EXAMPLES")
    print("="*70 + "\n")
    
    real_examples = [
        ["who", "you"],
        ["what", "name"],
        ["how", "you"],
        ["where", "you", "go"],
        ["thank", "you"],
        ["hello"],
        ["help", "me"],
        ["i", "eat"],
        ["you", "come", "home"],
    ]
    
    for signs in real_examples:
        sequence_str = " → ".join(signs)
        corrected = corrector.correct_sentence(" ".join(signs))
        
        print(f"Signs detected: {sequence_str}")
        print(f"Sentence:       {corrected}")
        print()
    
    print("="*70)
    print("\nHow it works:")
    print("1. System detects individual signs: 'who', 'you'")
    print("2. Grammar corrector recognizes pattern")
    print("3. Adds missing words: 'who' + 'are' + 'you'")
    print("4. Adds punctuation: 'Who are you?'")
    print("5. Text-to-speech speaks: 'Who are you?'")
    print("\nThis makes the translation natural and grammatically correct!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_phrase_construction()
