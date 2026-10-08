# ===============================
# Sentiment Classification using BERT
# ===============================

# Install dependencies:
# pip install transformers torch

from transformers import pipeline

# Step 1: Load pre-trained BERT sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Step 2: Input Sentences
sentences = [
    "I love using natural language processing!",
    "This is the worst movie I have ever watched.",
    "The food was okay, nothing special."
]

# Step 3: Perform Sentiment Classification
results = classifier(sentences)

# Step 4: Print Results
for sentence, result in zip(sentences, results):
    print(f"Sentence: {sentence}")
    print(f"Label: {result['label']}, Confidence: {result['score']:.4f}")
    print()

Input:
I love using natural language processing!
This is the worst movie I have ever watched.
The food was okay, nothing special.