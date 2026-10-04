import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# Download NLTK resources (only first time)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)

# Input text
text = "Barack Obama was the 44th President of the United States. He was born in Hawaii."

# Step 1: Tokenize text
tokens = word_tokenize(text)

# Step 2: POS Tagging
pos_tags = pos_tag(tokens)
print("POS Tagging Result:")
print(pos_tags)

# Step 3: Named Entity Recognition
ner_tree = ne_chunk(pos_tags)
print("\nNamed Entity Recognition Result:")
print(ner_tree)

# Extract named entities from tree
named_entities = []
for subtree in ner_tree:
    if type(subtree) == Tree:  # If subtree is a named entity
        entity_name = " ".join([token for token, pos in subtree.leaves()])
        entity_type = subtree.label()
        named_entities.append((entity_name, entity_type))

print("\nExtracted Named Entities:")
print(named_entities)
