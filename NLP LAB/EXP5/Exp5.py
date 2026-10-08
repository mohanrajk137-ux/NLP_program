import nltk
from nltk.tag import hmm
from nltk.corpus import treebank

# Download required NLTK resources
nltk.download('treebank')
nltk.download('universal_tagset')
# Load a portion of the Penn Treebank corpus
train_data = treebank.tagged_sents(tagset='universal')[:3000]
test_data = treebank.tagged_sents(tagset='universal')[3000:]
# Train an HMM model
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)
# Test the model
accuracy = hmm_tagger.evaluate(test_data)
print(f"HMM Tagger Accuracy: {accuracy:.4f}")

# Test with a sentence
sentence = "The quick brown fox jumps over the lazy dog".split()
print("Tagged Sentence:")
print(hmm_tagger.tag(sentence))