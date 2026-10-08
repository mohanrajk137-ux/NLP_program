# ===============================
# Dependency Parsing using spaCy
# ===============================

# Install: pip install spacy
# Download model: python -m spacy download en_core_web_sm

import spacy
from spacy import displacy

# Step 1: Load English model
nlp = spacy.load("en_core_web_sm")

# Step 2: Input Sentence
sentence = "The quick brown fox jumps over the lazy dog."
doc = nlp(sentence)

# Step 3: Print dependency relations
print("Token\tHead\tPOS\tDependency")
for token in doc:
    print(f"{token.text}\t{token.head.text}\t{token.pos_}\t{token.dep_}")

# Step 4: Visualize dependency structure in browser
displacy.serve(doc, style="dep")

Input:
The quick brown fox jumps over the lazy dog.