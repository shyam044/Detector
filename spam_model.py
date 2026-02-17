import pandas as pd
import numpy as np
import string
import nltk
import ssl
import pickle
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional, Dropout, Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential  # Sequential is moved here
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Required downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except:
    print("NLTK data already present or Offline.")

# 1. Load and Clean Data
data = pd.read_csv('email.csv')
data['text'] = data['text'].str.replace('Subject:', '', case=False)

# 2. Advanced Preprocessing (Lemmatization instead of Stemming)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuations_list = string.punctuation

def clean_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', punctuations_list))
    # Lemmatize and remove stopwords
    words = text.lower().split()
    clean_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(clean_words)

data['text'] = data['text'].apply(clean_text)

# 3. Train/Test Split
train_X, test_X, train_Y, test_Y = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Convert labels to numeric
train_Y = (train_Y == 'spam').astype(int)
test_Y = (test_Y == 'spam').astype(int)

# 4. Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)

max_len = 150 
train_sequences = pad_sequences(tokenizer.texts_to_sequences(train_X), maxlen=max_len, padding='post')
test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_X), maxlen=max_len, padding='post')

# 5. Handle Imbalance using Class Weights (Better than Oversampling)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_Y),
    y=train_Y
)
class_weights_dict = {i: weights[i] for i in range(len(weights))}

# 6. Model Architecture
# CORRECTED MODEL BLOCK
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Training
es = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)

model.fit(
    train_sequences, train_Y,
    validation_data=(test_sequences, test_Y),
    epochs=10,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[es]
)

# 8. Optimized Prediction Function
def predict_spam(email_text):
    processed_text = clean_text(email_text.replace('Subject:', ''))
    seq = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    
    prediction = model.predict(padded)[0][0]
    
    # Use a higher threshold (0.7) for financial accuracy
    return 'spam' if prediction > 0.7 else 'ham'

# Save assets
model.save("model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model retrained and saved successfully.")
