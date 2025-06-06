import numpy as np
import pandas as pd
import textstat
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import joblib
from scipy.sparse import hstack


# Load pre-trained tokenizer & model for perplexity calculation
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained("/work/ai_detector_backend/xgbc/gpt2_local") #add /work/ for docker
model = GPT2LMHeadModel.from_pretrained(model_name).to("cpu")  # Use GPU if available
model.eval()


with open("/work/ai_detector_backend/xgbc/stopwords.txt", "r") as f:
    stop_words = [line.strip() for line in f if line.strip()]
f.close()

# Load the vectorizer and encoder
vectorizer = joblib.load("/work/ai_detector_backend/xgbc/tfidf_vectorizer.pkl")
encoder = joblib.load("/work/ai_detector_backend/xgbc/label_encoder.pkl")


def calculate_perplexity(text):
    """Computes the perplexity of a given text using GPT-2."""
    inputs = tokenizer.encode(text, return_tensors='pt').to("cpu")

    if inputs.shape[1] > 1024:  # Truncate long texts
        inputs = inputs[:, :1024]

    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        log_likelihood = outputs.loss.item() * inputs.size(1)

    return np.exp(log_likelihood / inputs.size(1))

def extract_features(text):
    """Extracts all required features from input text."""
    words = text.split()
    word_count = len(words)
    sentence_count = textstat.sentence_count(text)

    features = {
        "text_length": len(text),
        "word_count": word_count,
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "sentence_count": sentence_count,
        "syllable_count": textstat.syllable_count(text),
        "avg_word_length": sum(len(word) for word in words) / word_count if word_count else 0,
        "perplexity": calculate_perplexity(text),
        "char_count": len(text),
        "unique_word_count": len(set(words)),
        "unique_word_ratio": len(set(words)) / (word_count + 1),  # +1 to avoid division by zero
        "stopword_count": sum(1 for w in words if w.lower() in stop_words),
        "burstiness": word_count / (sentence_count + 1),  # Variation in sentence length
        "entropy": -sum((words.count(w) / word_count) * np.log2(words.count(w) / word_count) for w in set(words) if word_count),  # Lexical diversity
    }

    return features

def preprocess_and_vectorize(text):
    """Preprocess the text, extract features, and transform it into a model-ready format."""
    # Extract handcrafted features
    feature_dict = extract_features(text)

    # Convert to DataFrame
    feature_df = pd.DataFrame([feature_dict])

    # TF-IDF Transformation
    tfidf_features = vectorizer.transform([text])

    # Combine handcrafted features with TF-IDF features
    final_features = hstack([tfidf_features, feature_df.values])

    return final_features
