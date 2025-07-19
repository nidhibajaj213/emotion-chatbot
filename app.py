import streamlit as st
from textblob import TextBlob
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

st.title("Emotion Detection Chatbot 🤖")

@st.cache_resource(show_spinner=True)
def load_models():
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
    return emotion_classifier, model, tokenizer

try:
    emotion_classifier, model, tokenizer = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

def get_sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Returns a float between -1 and 1

def detect_emotion(text):
    result = emotion_classifier(text)
    return result[0]['label']

def chat_with_bot(input_text):
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response.strip()

def chat_with_bot_and_emotion(input_text):
    response = chat_with_bot(input_text)
    sentiment_score = get_sentiment_score(input_text)
    emotion = detect_emotion(input_text)

    if sentiment_score > 0.05:
        response = f"Glad you're in a good mood! 😊 {response}"
    elif sentiment_score < -0.05:
        response = f"I'm sorry to hear you're feeling down. 😔 {response}"
    else:
        response = f"Thanks for sharing! {response}"

    # Append emotion-specific guidance only if it matches the context
    if emotion == 'anger' and sentiment_score < 0:
        response += "\nIt seems like you're feeling angry. Let me know how I can help!"
    elif emotion == 'fear' and sentiment_score < 0:
        response += "\nYou might be feeling anxious. I'm here to talk it out."
    elif emotion == 'joy' and sentiment_score >= 0:
        response += "\nYou're feeling joyful! 😊 Let's keep it going."
    elif emotion == 'sadness' and sentiment_score < 0:
        response += "\nIt sounds like you're feeling a bit sad. If you need someone to talk to, I'm here."

    return response

# --- Streamlit Interaction ---
user_input = st.text_input("You:", placeholder="How are you feeling today?")

if user_input:
    with st.spinner("Bot is thinking..."):
        bot_response = chat_with_bot_and_emotion(user_input)
    st.success(f"**Bot:** {bot_response}")
