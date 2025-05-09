from textblob import TextBlob

import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load pre-trained emotion detection model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Range: -1 (negative) to 1 (positive)
    if sentiment > 0:
        return 'happy'
    elif sentiment < 0:
        return 'sad'
    else:
        return 'neutral'

def detect_emotion(text):
    result = emotion_classifier(text)
    emotion = result[0]['label']
    return emotion

def chat_with_bot(input_text):
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

def chat_with_bot_and_emotion(input_text):
    response = chat_with_bot(input_text)
    sentiment = get_sentiment(input_text)
    emotion = detect_emotion(input_text)

    if sentiment == "happy":
        response = f"Glad you're in a good mood! 😊 {response}"
    elif sentiment == "sad":
        response = f"I'm sorry to hear you're feeling down. 😔 {response}"
    else:
        response = f"Thanks for sharing! {response}"

    if emotion == 'anger':
        response += "\nIt seems like you're feeling angry. Let me know how I can help!"
    elif emotion == 'fear':
        response += "\nYou might be feeling anxious. I'm here to talk it out."
    elif emotion == 'joy':
        response += "\nYou're feeling joyful! 😊 Let's keep it going."
    elif emotion == 'sadness':
        response += "\nIt sounds like you're feeling a bit sad. If you need someone to talk to, I'm here."

    return response

# Title of the web app
st.title("Emotion Detection Chatbot")

# Input field for user
user_input = st.text_input("Say something:")

# Generate response
if user_input:
    bot_response = chat_with_bot_and_emotion(user_input)
    st.write(f"Bot: {bot_response}")
