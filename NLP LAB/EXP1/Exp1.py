import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Function to perform basic text processing operations
def text_processing(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Sentence segmentation
    sentences = sent_tokenize(text)
    
    # Stopword removal
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return tokens, sentences, filtered_tokens, stemmed_tokens, lemmatized_tokens

# Example usage
text = "This is an example sentence. It demonstrates basic text processing operations."
tokens, sentences, filtered_tokens, stemmed_tokens, lemmatized_tokens = text_processing(text)

print("Original Text:")
print(text)
print("")

print("Tokens:")
print(tokens)
print("")

print("Sentences:")
print(sentences)
print("")

print("Filtered Tokens (after stopword removal):")
print(filtered_tokens)
print("")

print("Stemmed Tokens:")
print(stemmed_tokens)
print("")

print("Lemmatized Tokens:")
print(lemmatized_tokens)