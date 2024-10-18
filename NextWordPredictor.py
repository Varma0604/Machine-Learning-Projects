import torch
import re
import spacy
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, classification_report

# Load Spacy model (large model for better performance)
nlp = spacy.load('en_core_web_lg')

# Precompile regex for URL, repeated characters, and words
url_pattern = re.compile(r'http\S+|www\S+|https\S+')
repeated_char_pattern = re.compile(r'(.)\1{3,}')
repeated_word_pattern = re.compile(r'\b(\w+)( \1\b)+')

# Custom stopwords list, keeping important negation words
stopwords_to_keep = {"not", "no", "nor"}
custom_stopwords = nlp.Defaults.stop_words - stopwords_to_keep

# Function to remove repeated characters and words
def remove_gibberish(text):
    text = repeated_char_pattern.sub(r'\1', text)
    text = repeated_word_pattern.sub(r'\1', text)
    return text

# Additional feature: text normalization (converting to lowercase, handling contractions, etc.)
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"’", "'", text)  # Convert fancy apostrophes to simple ones
    text = re.sub(r"can't", "cannot", text)  # Example: expand "can't" to "cannot"
    text = re.sub(r"n't", " not", text)  # Handle "n't" contractions
    text = re.sub(r"'re", " are", text)  # Handle "'re" contractions
    return text

# Enhanced text preprocessing function with lemmatization, stopword removal, and normalization
def preprocess_text(text):
    text = url_pattern.sub('', text)  # Remove URLs
    text = remove_gibberish(text)     # Remove repeated characters/words
    text = normalize_text(text)       # Normalize text
    
    doc = nlp(text)                   # Apply Spacy for tokenization and lemmatization
    
    # Lemmatize, remove stopwords, punctuation, digits, long tokens, and tokens in custom stopwords
    tokens = [token.lemma_.lower() for token in doc if not (
        token.is_punct or token.is_digit or len(token.text) > 20 or token.text in custom_stopwords)]
    
    # Join tokens back into clean text
    clean_text = " ".join(tokens)
    
    return clean_text

# Parallelized preprocessing for large datasets
def combined_preprocessing(text_series):
    results = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(preprocess_text)(text) for text in tqdm(text_series, desc="Processing texts")
    )
    return pd.Series(results)

# Define a pipeline with custom preprocessing step
preprocess_pipe = Pipeline([
    ('preprocess', FunctionTransformer(combined_preprocessing, validate=False)),
])

# --- BERT Tokenization and Inference ---

# Load pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # Assuming binary classification (adjust num_labels as needed)

# Function to preprocess text for BERT tokenization
def preprocess_for_bert(text_list, max_len):
    input_ids = []
    attention_masks = []

    # Tokenize each sentence and create attention masks
    for text in text_list:
        encoded_dict = tokenizer.encode_plus(
            text,                      # Text to encode
            add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
            max_length=max_len,         # Pad or truncate to max length
            padding='max_length',       # Use padding up to max_length
            return_attention_mask=True, # Create attention mask
            return_tensors='pt',        # Return PyTorch tensors
            truncation=True             # Truncate if the sentence is too long
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_masks

# Function to perform inference using the BERT model and return predictions with confidence scores
def predict(text_list, max_len, device):
    # Step 1: Apply custom preprocessing
    preprocessed_texts = preprocess_pipe.transform(pd.Series(text_list))
    
    # Step 2: Convert preprocessed text into BERT input format
    input_ids, attention_masks = preprocess_for_bert(preprocessed_texts, max_len)

    # Move inputs to the correct device (CPU or GPU)
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)

    # Perform inference without gradient calculation
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
    
    # Extract logits (raw model outputs) and calculate softmax for confidence scores
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # Get predicted class labels (highest probability)
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    
    # Get confidence scores for the predicted class
    confidence_scores = probabilities.max(dim=1).values.cpu().numpy()
    
    return predictions, confidence_scores

# --- Example Usage ---

# Sample text list
sample_texts = ["I love this product! It works perfectly.", 
                "Terrible customer service. Never buying again."]

# Set device (use GPU if available)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Predict class and get confidence scores
predictions, confidences = predict(sample_texts, max_len=128, device=device)

# Output predictions and their confidence levels
for text, pred, conf in zip(sample_texts, predictions, confidences):
    print(f"Text: {text}\nPrediction: {pred}, Confidence: {conf:.4f}\n")

