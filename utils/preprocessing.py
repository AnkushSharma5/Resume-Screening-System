import re
import nltk
import spacy
from nltk.corpus import stopwords

# Ensure NLTK stopwords are available
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """
    Cleans and preprocesses resume text.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    doc = nlp(" ".join(tokens))

    lemmatized_tokens = [token.lemma_ for token in doc]

    # print(lemmatized_tokens)

    return " ".join(lemmatized_tokens)