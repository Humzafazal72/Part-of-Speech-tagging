import pickle
import numpy as np
import streamlit as st

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("POS Tagging App")

# Load necessary files
with open('POS_tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

with open('enc.pkl', 'rb') as file:
    pos = pickle.load(file)

# Swapping POS to map back to labels
swapped_pos = {v: k for k, v in pos.items()}

# Colors for each POS (39 different colors)
colors = [
    "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#6A5ACD", "#FF69B4", 
    "#8B0000", "#00CED1", "#FF4500", "#2E8B57", "#DAA520", "#9932CC", 
    "#FF1493", "#B22222", "#00FA9A", "#B8860B", "#BA55D3", "#FF7F50", 
    "#66CDAA", "#FF6347", "#483D8B", "#40E0D0", "#D2691E", "#7B68EE", 
    "#C71585", "#228B22", "#FFDAB9", "#CD5C5C", "#87CEEB", "#32CD32", 
    "#FFA500", "#FF0000", "#8A2BE2", "#DDA0DD", "#00FF7F", "#4682B4", 
    "#A52A2A", "#7FFF00", "#DA70D6"
]

# Cleaning the POS and text column
def clean_input(sentence):
    punctuation = "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'"
    omit = ["'s", "'ll", "--", '"', '..']
    sentence = sentence.lower()
    words = sentence.split()
    clean_sentence = [word for word in words if word not in punctuation and word not in omit and not word.isdecimal() and not word.isnumeric()]
    return clean_sentence

# Preprocessing input for the model
def preprocess_input(clean_sentence):
    sequences = tokenizer.texts_to_sequences(clean_sentence)
    sequences = np.array(sequences).reshape(-1,)
    pad_seq = pad_sequences([sequences], maxlen=30, padding='post').reshape(-1, 1)
    return pad_seq

# Processing the predictions
def process_pred(result, input_):
    pos_ = []
    for each in result:
        m = 0
        ind = 0
        for i in range(len(each[0])):
            try:
                if each[0][i] > m:
                    m = each[0][i]
                    ind = i
            except:
                print(each[i])
        pos_.append(ind)
    
    POS = [swapped_pos[word] for word in pos_]
    len_ = len(input_)
    return POS[:len_]

# Function to display words with colored POS tags
def display_with_color(words, pos_tags):
    colored_text = ""
    for i, (word, tag) in enumerate(zip(words, pos_tags)):
        color = colors[i % len(colors)]  # Cycle through colors if tags exceed 39
        colored_text += f'<span style="color:{color}; font-weight:bold">{word} ({tag})</span> '
    
    st.markdown(f"<p>{colored_text}</p>", unsafe_allow_html=True)

# Input: Accept string with a word count less than or equal to 30
input_text = st.text_area("Enter the text (Max 30 words):", "")
button = st.button(label='Submit')
if input_text and button:
    words = input_text.split()
    word_count = len(words)
    if word_count > 30:
        st.error("The input exceeds the 30-word limit. Please enter a shorter text.")
    else:
        # Clean and preprocess the input text
        clean_sentence = clean_input(input_text)
        processed_input = preprocess_input(clean_sentence)

        # Load LSTM model
        lstm = load_model('lstm_model.h5')
        pred = lstm.predict(processed_input)
        
        # Process prediction to get POS tags
        pos_tags = process_pred(pred, clean_sentence)
        
        # Display the words along with colored POS tags
        display_with_color(clean_sentence, pos_tags)
