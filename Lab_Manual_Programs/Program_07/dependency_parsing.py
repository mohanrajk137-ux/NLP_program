import sys
import re

# Python 3.14 compatibility fix for pydantic v1 used by spaCy
try:
    import pydantic.v1.fields
    import pydantic.v1.schema

    orig_set_default = pydantic.v1.fields.ModelField._set_default_and_type
    def patched_set_default(self):
        try:
            return orig_set_default(self)
        except Exception:
            self.type_ = str
            self.outer_type_ = str
    pydantic.v1.fields.ModelField._set_default_and_type = patched_set_default

    orig_ann = pydantic.v1.schema.get_annotation_from_field_info
    def patched_ann(annotation, field_info, name, validate_assignment):
        try:
            return orig_ann(annotation, field_info, name, validate_assignment)
        except ValueError:
            return annotation
    pydantic.v1.schema.get_annotation_from_field_info = patched_ann
except ImportError:
    pass

import spacy
from spacy import displacy

# Step 1: Load English model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Step 2: Input Sentence
sentence = "The quick brown fox jumps over the lazy dog."
doc = nlp(sentence)

# Step 3: Print dependency relations
print("Token\tHead\tPOS\tDependency")
for token in doc:
    print(f"{token.text}\t{token.head.text}\t{token.pos_}\t{token.dep_}")

# Step 4: Visualize dependency structure in browser (commented or optional)
# displacy.serve(doc, style="dep")
