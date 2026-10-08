import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
# Download NLTK resources (only first time)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
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

Input Text:
Barack Obama was the 44th President of the United States. He was born in Hawaii.