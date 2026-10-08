import nltk
from nltk.util import ngrams
from collections import Counter
nltk.download('punkt')
# Sample text
text = "I love natural language processing and I love Python programming."
# Tokenize text into words
words = nltk.word_tokenize(text.lower())
# Create bigrams (N=2)
bigrams = list(ngrams(words, 2))
# Count bigram frequencies
bigram_freq = Counter(bigrams)
print("Bigrams and their frequencies:")
for bigram, freq in bigram_freq.items():
    print(f"{bigram} -> {freq}")

def predict_next_word(word, bigram_freq):
    candidates = {w2: freq for (w1, w2), freq in bigram_freq.items() if w1 == word}
    if not candidates:
        return None
    return max(candidates, key=candidates.get) 
 # word with max frequency
# Example prediction
current_word = "love"
next_word = predict_next_word(current_word, bigram_freq)
print(f"Next word after '{current_word}' is likely '{next_word}'")